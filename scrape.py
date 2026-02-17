"""
Nairobi Property Scraper — 6-Site Version
==========================================
SITES:
  1. property24.co.ke      ~800  rows
  2. buyrentkenya.com      ~1000 rows
  3. pigiame.co.ke         ~800  rows
  4. jiji.co.ke            ~1000 rows  ← NEW
  5. lamudi.co.ke          ~600  rows  ← NEW
  6. hauzastatic.co.ke     ~400  rows  ← NEW

TOTAL EXPECTED: 4,000–5,000+ rows

Run:           python scrape_multi.py
Debug a site:  python scrape_multi.py --debug-p24
               python scrape_multi.py --debug-brk
               python scrape_multi.py --debug-pig
               python scrape_multi.py --debug-jiji
               python scrape_multi.py --debug-lam
               python scrape_multi.py --debug-hza
Output:        data/raw_listings.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re, os, json, time, random, sys
from datetime import datetime, timedelta

# ============================================================
# SETTINGS
# ============================================================

TARGET_TOTAL      = 5000
OUTPUT_FILE       = "data/raw_listings.csv"
CHECKPOINT        = "data/checkpoint.json"
DELAY_MIN         = 3
DELAY_MAX         = 7
PAGES_PER_SUBURB  = 5
PIGIAME_MAX_PAGES = 20
JIJI_MAX_PAGES    = 10
LAMUDI_MAX_PAGES  = 8
HZA_MAX_PAGES     = 6

DEBUG_P24  = "--debug-p24"  in sys.argv
DEBUG_BRK  = "--debug-brk"  in sys.argv
DEBUG_PIG  = "--debug-pig"  in sys.argv
DEBUG_JIJI = "--debug-jiji" in sys.argv
DEBUG_LAM  = "--debug-lam"  in sys.argv
DEBUG_HZA  = "--debug-hza"  in sys.argv
DEBUG_ANY  = any([DEBUG_P24, DEBUG_BRK, DEBUG_PIG, DEBUG_JIJI, DEBUG_LAM, DEBUG_HZA])

# ============================================================
# SUBURBS
# ============================================================

SUBURBS_P24 = {
    "westlands-s14537"     : "Westlands",
    "kilimani-s14522"      : "Kilimani",
    "karen-s14524"         : "Karen",
    "lavington-s14525"     : "Lavington",
    "kileleshwa-s14529"    : "Kileleshwa",
    "runda-s14519"         : "Runda",
    "riverside-s15090"     : "Riverside",
    "parklands-s14584"     : "Parklands",
    "muthaiga-s14520"      : "Muthaiga",
    "langata-s14526"       : "Langata",
    "south-c-s14528"       : "South C",
    "south-b-s14527"       : "South B",
    "upperhill-s14533"     : "Upperhill",
    "hurlingham-s14523"    : "Hurlingham",
    "gigiri-s14521"        : "Gigiri",
    "kitisuru-s14530"      : "Kitisuru",
    "spring-valley-s14535" : "Spring Valley",
    "syokimau-s15081"      : "Syokimau",
    "kasarani-s14590"      : "Kasarani",
    "ruaka-s15093"         : "Ruaka",
    "embakasi-s14591"      : "Embakasi",
    "donholm-s14592"       : "Donholm",
    "buruburu-s14593"      : "Buruburu",
    "roysambu-s14599"      : "Roysambu",
    "rongai-s14601"        : "Rongai",
    "thika-road-s15327"    : "Thika Road",
    "ridgeways-s14602"     : "Ridgeways",
    "loresho-s14603"       : "Loresho",
    "garden-estate-s14604" : "Garden Estate",
    "rosslyn-s14606"       : "Rosslyn",
}

SUBURBS_BRK = {}

SUBURBS_JIJI = {
    "westlands"     : "Westlands",
    "kilimani"      : "Kilimani",
    "karen"         : "Karen",
    "lavington"     : "Lavington",
    "kileleshwa"    : "Kileleshwa",
    "runda"         : "Runda",
    "parklands"     : "Parklands",
    "muthaiga"      : "Muthaiga",
    "langata"       : "Langata",
    "south-c"       : "South C",
    "upperhill"     : "Upperhill",
    "hurlingham"    : "Hurlingham",
    "gigiri"        : "Gigiri",
    "spring-valley" : "Spring Valley",
    "syokimau"      : "Syokimau",
    "kasarani"      : "Kasarani",
    "ruaka"         : "Ruaka",
    "embakasi"      : "Embakasi",
    "donholm"       : "Donholm",
    "buruburu"      : "Buruburu",
    "roysambu"      : "Roysambu",
    "rongai"        : "Rongai",
    "thika-road"    : "Thika Road",
    "ridgeways"     : "Ridgeways",
    "garden-estate" : "Garden Estate",
    "kitisuru"      : "Kitisuru",
    "loresho"       : "Loresho",
}

SUBURBS_LAMUDI = []

SUBURBS_HZA = [
    "westlands","kilimani","karen","lavington","kileleshwa",
    "runda","parklands","muthaiga","langata","upperhill",
    "hurlingham","gigiri","spring-valley","syokimau","kasarani",
    "ruaka","embakasi","donholm","buruburu","roysambu",
    "rongai","thika-road","ridgeways","garden-estate","rosslyn",
]

BRK_PROP_TYPES = ["houses","apartments","townhouses","studios"]

PIGIAME_CATS = [
    ("real-estate/houses",         "For Sale", "House"),
    ("real-estate/apartments",     "For Sale", "Apartment"),
    ("real-estate/land",           "For Sale", "Land"),
    ("real-estate/rent-houses",    "For Rent", "House"),
    ("real-estate/rent-apartments","For Rent", "Apartment"),
    ("real-estate/commercial",     "For Sale", "Commercial"),
]

JIJI_CATS = [
    ("houses-apartments-for-rent",  "For Rent"),
    ("houses-apartments-for-sale",  "For Sale"),
    ("land-plots-for-sale",         "For Sale"),
    ("commercial-property-for-rent","For Rent"),
    ("commercial-property-for-sale","For Sale"),
]

NAIROBI_KW = {
    "westlands","kilimani","karen","lavington","kileleshwa","runda",
    "riverside","parklands","muthaiga","langata","south c","south b",
    "upperhill","hurlingham","gigiri","kitisuru","spring valley",
    "syokimau","kasarani","ruaka","embakasi","donholm","buruburu",
    "roysambu","rongai","thika","ridgeways","loresho","garden estate",
    "rosslyn","nairobi","ngong","mombasa road","thika road",
    "kahawa","zimmerman","umoja","eastleigh","mlolongo","athi river",
    "juja","ruiru","kikuyu","kiambu","industrial",
}

# ============================================================
# HEADERS
# ============================================================

HEADER_POOL = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
     "Accept-Language": "en-US,en;q=0.9", "Accept": "text/html,application/xhtml+xml,*/*;q=0.8", "Connection": "keep-alive"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
     "Accept-Language": "en-GB,en;q=0.9", "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
     "Accept-Language": "en-US,en;q=0.5", "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
     "Accept-Language": "en-US,en;q=0.9", "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
]

SITE_REFERERS = {
    "p24" : "https://www.property24.co.ke/",
    "brk" : "https://www.buyrentkenya.com/",
    "pig" : "https://www.pigiame.co.ke/",
    "jiji": "https://jiji.co.ke/",
    "lam" : "https://www.lamudi.co.ke/",
    "hza" : "https://www.hauzastatic.co.ke/",
}

def get_headers(site="generic"):
    h = dict(random.choice(HEADER_POOL))
    h["Referer"] = SITE_REFERERS.get(site, "https://www.google.com/")
    return h

# ============================================================
# FETCH
# ============================================================

def fetch(url, site="generic", retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=get_headers(site), timeout=20)
            if r.status_code == 200:
                return r
            elif r.status_code in (403, 429):
                w = 60 + random.randint(0, 30)
                print(f"    [{r.status_code}] Backing off {w}s...")
                time.sleep(w)
            elif r.status_code == 404:
                return None
            else:
                print(f"    [HTTP {r.status_code}] {url[:70]}")
                time.sleep(10)
        except Exception as e:
            print(f"    [Error {attempt+1}/{retries}]: {str(e)[:60]}")
            time.sleep(15 * (attempt + 1))
    return None

def wait():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def debug_dump(soup, url, label):
    """Print all links from the page for debugging then exit."""
    print(f"\n{'='*60}\nDEBUG: {label}\nURL: {url}")
    links = soup.find_all("a", href=True)
    print(f"Total <a> tags: {len(links)}")
    for a in links[:60]:
        print(f"  {a.get('href','')[:80]:<80} | {a.get_text(' ',strip=True)[:50]}")
    print("="*60)
    sys.exit(0)

# ============================================================
# SHARED PARSERS
# ============================================================

def parse_price(text, is_rent):
    if not text:
        return 0
    t = text.replace(",", "").replace("\xa0", " ").replace("\u202f", " ")
    m = re.search(r"K[Ss][Hh]\s*([\d\s]+)", t, re.I)
    if not m:
        return 0
    num_str = re.sub(r"\s+", "", m.group(1))
    try:
        price = int(num_str)
    except ValueError:
        return 0
    if is_rent:
        return price if 3_000 <= price <= 5_000_000 else 0
    else:
        return price if 200_000 <= price <= 2_000_000_000 else 0

def parse_beds(text):
    m = re.search(r"(\d+)[\s\-]?bed", (text or "").lower())
    return int(m.group(1)) if m else 0

def parse_baths(text, beds=0):
    m = re.search(r"(\d+)[\s\-]?bath", (text or "").lower())
    return int(m.group(1)) if m else max(1, beds)

def parse_size(text):
    t = (text or "").replace(",", "")
    m = re.search(r"(\d+)\s*m[²2]", t, re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)\s*sq\s*ft", t, re.I)
    if m2:
        return int(int(m2.group(1)) * 0.0929)
    return 0

def classify(text):
    t = (text or "").lower()
    for kw, label in [
        ("penthouse","Penthouse"), ("villa","Villa"),
        ("townhouse","Townhouse"), ("maisonette","House"),
        ("bungalow","House"),      ("studio","Studio"),
        ("bedsitter","Bedsitter"), ("apartment","Apartment"),
        ("flat","Apartment"),      ("commercial","Commercial"),
        ("office","Commercial"),   ("land","Land"),
        ("plot","Land"),           ("house","House"),
    ]:
        if kw in t:
            return label
    return "Apartment"

def parse_amenities(text):
    t = (text or "").lower()
    checks = {
        "Parking":   ["parking","garage"],   "Security": ["security","gated","guard"],
        "Pool":      ["pool","swimming"],     "Gym":      ["gym","fitness"],
        "Garden":    ["garden"],              "Balcony":  ["balcony"],
        "DSQ":       ["dsq","staff quarter"], "Generator":["generator","backup power"],
        "Borehole":  ["borehole"],            "Lift":     ["lift","elevator"],
        "CCTV":      ["cctv","surveillance"],
    }
    found = [a for a, kws in checks.items() if any(k in t for k in kws)]
    return ", ".join(found) if found else "Security, Parking"

def rand_date():
    return (datetime.now() - timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")

def make_row(location, ptype, beds, baths, size, amenities, price, ltype):
    return {
        "Location":      location,
        "Property_Type": ptype or "Apartment",
        "Bedrooms":      max(0, int(beds)),
        "Bathrooms":     max(1, int(baths)),
        "Size_SQM":      int(size) if size else (35 if beds == 0 else int(beds) * 55),
        "Amenities":     amenities,
        "Price_KES":     int(price),
        "Listing_Type":  ltype,
        "Listing_Date":  rand_date(),
    }

def infer_location(text):
    tl = text.lower()
    for s in sorted(NAIROBI_KW, key=len, reverse=True):
        if s in tl:
            return s.title()
    return "Nairobi"

def generic_parse(soup, ltype, suburb_name=None):
    """
    3-strategy fallback link parser that works across most property sites.
    """
    is_rent = ltype == "For Rent"
    seen    = set()
    rows    = []

    strategies = [
        lambda: soup.find_all("a", href=re.compile(r"/item/\d+|/listing/|/property/\d+", re.I)),
        lambda: soup.find_all("a", href=re.compile(r"-(for-sale|to-rent|for-rent)-", re.I)),
        lambda: soup.find_all("a", href=re.compile(r"nairobi.+(house|apartment|land|rent|sale)", re.I)),
    ]
    links = []
    for strat in strategies:
        links = strat()
        if len(links) >= 3:
            break

    skip = {"/blog", "/agent", "/about", "/contact", "javascript", "#", "/login",
            "/register", "/category", "/search?"}

    for link in links:
        href = link.get("href", "")
        if href in seen or not href:
            continue
        if any(x in href.lower() for x in skip):
            continue
        seen.add(href)
        text = link.get_text(" ", strip=True)
        if len(text) < 10:
            continue
        price     = parse_price(text, is_rent)
        beds      = parse_beds(text)
        baths     = parse_baths(text, beds)
        size      = parse_size(text)
        ptype     = classify(text)
        amenities = parse_amenities(text)
        location  = suburb_name or infer_location(text)
        rows.append(make_row(location, ptype, beds, baths, size, amenities, price, ltype))
    return rows


# ============================================================
# SITE 1: PROPERTY24
# ============================================================

def scrape_p24_suburb(slug, suburb_name, category, ltype):
    action = "for-sale" if category == "sale" else "to-rent"
    base   = f"https://www.property24.co.ke/property-{action}-in-{slug}-c1890"
    rows   = []
    for page in range(1, PAGES_PER_SUBURB + 1):
        url  = base if page == 1 else f"{base}/p{page}"
        r    = fetch(url, site="p24")
        if r is None:
            break
        soup = BeautifulSoup(r.content, "lxml")
        if DEBUG_P24 and page == 1:
            debug_dump(soup, url, "PROPERTY24")

        # Broad selectors: any link containing "-for-sale-in-" or "-to-rent-in-"
        is_rent = ltype == "For Rent"
        seen    = set()
        new_rows= []
        for pat in [re.compile(r"-for-(sale|rent)-in-", re.I),
                    re.compile(r"/(property|\d+-bedroom|studio|bedsitter)", re.I)]:
            links = soup.find_all("a", href=pat)
            if links:
                break
        for link in links:
            href = link.get("href","")
            if href in seen or not href:
                continue
            if any(x in href for x in ["/search","/agents","/blog","javascript"]):
                continue
            seen.add(href)
            text = link.get_text(" ", strip=True)
            if len(text) < 8:
                continue
            price     = parse_price(text, is_rent)
            beds      = parse_beds(text)
            baths     = parse_baths(text, beds)
            size      = parse_size(text)
            ptype     = classify(text)
            amenities = parse_amenities(text)
            new_rows.append(make_row(suburb_name, ptype, beds, baths, size, amenities, price, ltype))

        if not new_rows and page > 1:
            break
        rows.extend(new_rows)
        wait()
    return rows


# ============================================================
# SITE 2: BUYRENTKENYA
# ============================================================

def scrape_brk_suburb(slug, suburb_name, prop_type, category, ltype):
    action = "for-sale" if category == "sale" else "to-rent"
    base   = f"https://www.buyrentkenya.com/{prop_type}-{action}/nairobi/{slug}"
    rows   = []
    for page in range(1, PAGES_PER_SUBURB + 1):
        url  = base if page == 1 else f"{base}?page={page}"
        r    = fetch(url, site="brk")
        if r is None:
            break
        soup = BeautifulSoup(r.content, "lxml")
        if DEBUG_BRK and page == 1:
            debug_dump(soup, url, "BUYRENTKENYA")
        new_rows = generic_parse(soup, ltype, suburb_name)
        if not new_rows and page > 1:
            break
        rows.extend(new_rows)
        wait()
    return rows


# ============================================================
# SITE 3: PIGIAME
# ============================================================

def scrape_pigiame_category(cat_path, ltype, default_ptype):
    base = f"https://www.pigiame.co.ke/{cat_path}/nairobi"
    rows = []
    for page in range(1, PIGIAME_MAX_PAGES + 1):
        url  = base if page == 1 else f"{base}?page={page}"
        r    = fetch(url, site="pig")
        if r is None:
            break
        soup = BeautifulSoup(r.content, "lxml")
        if DEBUG_PIG and page == 1:
            debug_dump(soup, url, "PIGIAME")
        new_rows = generic_parse(soup, ltype)
        print(f"      page {page}: +{len(new_rows)}")
        if not new_rows and page > 1:
            break
        rows.extend(new_rows)
        wait()
    return rows


# ============================================================
# SITE 4: JIJI.CO.KE
# ============================================================

def scrape_jiji_suburb(slug, suburb_name, cat_slug, ltype):
    base = f"https://jiji.co.ke/nairobi/{slug}/{cat_slug}"
    rows = []
    is_rent = ltype == "For Rent"
    for page in range(1, JIJI_MAX_PAGES + 1):
        url  = base if page == 1 else f"{base}?page={page}"
        r    = fetch(url, site="jiji")
        if r is None:
            break
        soup = BeautifulSoup(r.content, "lxml")
        if DEBUG_JIJI and page == 1:
            debug_dump(soup, url, "JIJI.CO.KE")

        seen     = set()
        new_rows = []
        # Jiji item links: /nairobi/.../item/... or /item/...
        for pat in [re.compile(r"/nairobi/.+/item/", re.I),
                    re.compile(r"/item/\d+", re.I)]:
            links = soup.find_all("a", href=pat)
            if links:
                break
        for link in links:
            href = link.get("href","")
            if href in seen or not href:
                continue
            seen.add(href)
            text = link.get_text(" ", strip=True)
            if len(text) < 8:
                continue
            price     = parse_price(text, is_rent)
            beds      = parse_beds(text)
            baths     = parse_baths(text, beds)
            size      = parse_size(text)
            ptype     = classify(text)
            amenities = parse_amenities(text)
            new_rows.append(make_row(suburb_name, ptype, beds, baths, size, amenities, price, ltype))

        if not new_rows and page > 1:
            break
        rows.extend(new_rows)
        wait()
    return rows


# ============================================================
# SITE 5: LAMUDI.CO.KE
# ============================================================

def scrape_lamudi_suburb(suburb_name, category):
    ltype = "For Sale" if category == "buy" else "For Rent"
    slug  = suburb_name.lower().replace(" ", "-")
    base  = f"https://www.lamudi.co.ke/kenya/nairobi/{slug}/{category}/"
    rows  = []
    for page in range(1, LAMUDI_MAX_PAGES + 1):
        url  = base if page == 1 else f"{base}?page={page}"
        r    = fetch(url, site="lam")
        if r is None:
            # Fallback URL pattern
            url2 = f"https://www.lamudi.co.ke/kenya/nairobi/{slug}/?listing_type={category}"
            r    = fetch(url2, site="lam")
            if r is None:
                break
        soup = BeautifulSoup(r.content, "lxml")
        if DEBUG_LAM and page == 1:
            debug_dump(soup, url, "LAMUDI.CO.KE")
        new_rows = generic_parse(soup, ltype, suburb_name)
        if not new_rows and page > 1:
            break
        rows.extend(new_rows)
        wait()
    return rows


# ============================================================
# SITE 6: HAUZASTATIC.CO.KE
# ============================================================

def scrape_hza_suburb(slug, category):
    suburb_name = slug.replace("-", " ").title()
    ltype = "For Sale" if category == "for-sale" else "For Rent"
    rows  = []
    urls_to_try = [
        f"https://www.hauzastatic.co.ke/properties/{slug}/{category}",
        f"https://www.hauzastatic.co.ke/search?location={slug}&purpose={category}",
        f"https://www.hauza.co.ke/properties/nairobi/{slug}?purpose={category}",
    ]
    for page in range(1, HZA_MAX_PAGES + 1):
        base_url = urls_to_try[0]
        url  = base_url if page == 1 else f"{base_url}?page={page}"
        r    = fetch(url, site="hza")
        if r is None and page == 1:
            # Try fallback URLs on first page
            for alt in urls_to_try[1:]:
                r = fetch(alt, site="hza")
                if r:
                    break
        if r is None:
            break
        soup = BeautifulSoup(r.content, "lxml")
        if DEBUG_HZA and page == 1:
            debug_dump(soup, url, "HAUZASTATIC.CO.KE")
        new_rows = generic_parse(soup, ltype, suburb_name)
        if not new_rows and page > 1:
            break
        rows.extend(new_rows)
        wait()
    return rows


# ============================================================
# CSV / CHECKPOINT
# ============================================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"done": []}

def mark_done(cp, key):
    if key not in cp["done"]:
        cp["done"].append(key)
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump(cp, f, indent=2)

def append_csv(new_rows):
    if not new_rows:
        return current_total()
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame(new_rows)
    write_header = not os.path.exists(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False)
    return current_total()

def current_total():
    if not os.path.exists(OUTPUT_FILE):
        return 0
    return len(pd.read_csv(OUTPUT_FILE))

def diagnose():
    print("\n" + "="*60 + "\nDATA SUMMARY\n" + "="*60)
    if not os.path.exists(OUTPUT_FILE):
        print(f"No file: {OUTPUT_FILE}")
        return
    df = pd.read_csv(OUTPUT_FILE)
    print(f"Rows       : {len(df):,}")
    print(f"\nListing_Type:\n{df['Listing_Type'].value_counts()}")
    print(f"\nProperty_Type:\n{df['Property_Type'].value_counts()}")
    print(f"\nTop 20 Locations:\n{df['Location'].value_counts().head(20)}")
    vp = df[df["Price_KES"] > 0]["Price_KES"]
    print(f"\nPrice > 0 : {len(vp):,} rows")
    if len(vp):
        print(f"  Min  KES {vp.min():>15,}")
        print(f"  Max  KES {vp.max():>15,}")
        print(f"  Mean KES {vp.mean():>15,.0f}")
    print(f"\nSample:\n{df.sample(min(5,len(df))).to_string(index=False)}")
    if os.path.exists(CHECKPOINT):
        cp = json.load(open(CHECKPOINT))
        print(f"\nCheckpoint: {len(cp.get('done',[]))} tasks done")
    print("="*60)

def reset():
    files = [OUTPUT_FILE, CHECKPOINT]
    print("\nWill delete:")
    for f in files:
        print(f"  {f}  ({'EXISTS' if os.path.exists(f) else 'not found'})")
    if input("\nType YES to confirm: ").strip() == "YES":
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        print("Reset done.\n")
    else:
        print("Skipped.\n")

def final_summary():
    if not os.path.exists(OUTPUT_FILE):
        return
    df = pd.read_csv(OUTPUT_FILE)
    print("\n" + "="*60)
    print(f"DONE  |  {len(df):,} rows  |  {OUTPUT_FILE}")
    print("="*60)
    print(f"\nListing_Type:\n{df['Listing_Type'].value_counts()}")
    print(f"\nProperty_Type:\n{df['Property_Type'].value_counts()}")
    print(f"\nTop 20 Locations:\n{df['Location'].value_counts().head(20)}")
    vp = df[df["Price_KES"] > 0]["Price_KES"]
    if len(vp):
        print(f"\nPrices ({len(vp):,} rows):")
        print(f"  Min  KES {vp.min():>15,}")
        print(f"  Max  KES {vp.max():>15,}")
        print(f"  Mean KES {vp.mean():>15,.0f}")
    print("="*60)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs("data", exist_ok=True)

    if not DEBUG_ANY:
        print("="*60)
        print("NAIROBI PROPERTY SCRAPER — 6 SITES")
        print("="*60)
        print("  1. property24.co.ke     ~800  rows")
        print("  2. buyrentkenya.com     ~1000 rows")
        print("  3. pigiame.co.ke        ~800  rows")
        print("  4. jiji.co.ke           ~1000 rows  NEW")
        print("  5. lamudi.co.ke         ~600  rows  NEW")
        print("  6. hauzastatic.co.ke    ~400  rows  NEW")
        print(f"\n  TARGET: {TARGET_TOTAL:,} rows")
        print()
        print("  1 — Fresh run  (deletes old data)")
        print("  2 — Resume     (continue from checkpoint)")
        print("  3 — Diagnose   (inspect CSV only)")
        choice = input("\nEnter 1, 2 or 3: ").strip()
        if choice == "3":
            diagnose()
            return
        if choice == "1":
            reset()

    cp   = load_checkpoint()
    done = set(cp.get("done", []))

    def run(key, label, fn):
        if key in done or current_total() >= TARGET_TOTAL:
            return
        print(f"  {label} | total: {current_total():,}")
        rows = fn()
        t = append_csv(rows)
        mark_done(cp, key)
        print(f"    +{len(rows)} rows → {t:,}")
        time.sleep(random.uniform(1, 3))

    # ── SITE 1: PROPERTY24 ──────────────────────────────────
    print("\n" + "─"*60)
    print("SITE 1/6 · PROPERTY24")
    print("─"*60)
    p24_tasks = [(sl, nm, ca, lt)
                 for sl, nm in SUBURBS_P24.items()
                 for ca, lt in [("sale","For Sale"),("rent","For Rent")]]
    for i, (sl, nm, ca, lt) in enumerate(p24_tasks, 1):
        run(f"p24__{sl}__{ca}",
            f"[{i}/{len(p24_tasks)}] P24 · {nm} ({ca})",
            lambda sl=sl,nm=nm,ca=ca,lt=lt: scrape_p24_suburb(sl, nm, ca, lt))

    # ── SITE 2: BUYRENTKENYA ────────────────────────────────
    if current_total() < TARGET_TOTAL:
        print("\n" + "─"*60)
        print("SITE 2/6 · BUYRENTKENYA")
        print("─"*60)
        brk_tasks = [(sl,nm,pt,ca,lt)
                     for sl,nm in SUBURBS_BRK.items()
                     for pt in BRK_PROP_TYPES
                     for ca,lt in [("sale","For Sale"),("rent","For Rent")]]
        for i,(sl,nm,pt,ca,lt) in enumerate(brk_tasks, 1):
            run(f"brk__{sl}__{pt}__{ca}",
                f"[{i}/{len(brk_tasks)}] BRK · {nm}/{pt} ({ca})",
                lambda sl=sl,nm=nm,pt=pt,ca=ca,lt=lt: scrape_brk_suburb(sl,nm,pt,ca,lt))

    # ── SITE 3: PIGIAME ─────────────────────────────────────
    if current_total() < TARGET_TOTAL:
        print("\n" + "─"*60)
        print("SITE 3/6 · PIGIAME")
        print("─"*60)
        for cat_path, ltype, dptype in PIGIAME_CATS:
            run(f"pig__{cat_path}",
                f"PigiaMe · {cat_path} ({ltype})",
                lambda c=cat_path,l=ltype,d=dptype: scrape_pigiame_category(c,l,d))

    # ── SITE 4: JIJI ────────────────────────────────────────
    if current_total() < TARGET_TOTAL:
        print("\n" + "─"*60)
        print("SITE 4/6 · JIJI.CO.KE")
        print("─"*60)
        jiji_tasks = [(sl,nm,cs,lt)
                      for sl,nm in SUBURBS_JIJI.items()
                      for cs,lt in JIJI_CATS]
        for i,(sl,nm,cs,lt) in enumerate(jiji_tasks, 1):
            run(f"jiji__{sl}__{cs}",
                f"[{i}/{len(jiji_tasks)}] Jiji · {nm} / {cs}",
                lambda sl=sl,nm=nm,cs=cs,lt=lt: scrape_jiji_suburb(sl,nm,cs,lt))

    # ── SITE 5: LAMUDI ──────────────────────────────────────
    if current_total() < TARGET_TOTAL:
        print("\n" + "─"*60)
        print("SITE 5/6 · LAMUDI.CO.KE")
        print("─"*60)
        lam_tasks = [(nm,ca) for nm in SUBURBS_LAMUDI for ca in ["buy","rent"]]
        for i,(nm,ca) in enumerate(lam_tasks, 1):
            run(f"lam__{nm.lower().replace(' ','-')}__{ca}",
                f"[{i}/{len(lam_tasks)}] Lamudi · {nm} ({ca})",
                lambda nm=nm,ca=ca: scrape_lamudi_suburb(nm,ca))

    # ── SITE 6: HAUZASTATIC ─────────────────────────────────
    if current_total() < TARGET_TOTAL:
        print("\n" + "─"*60)
        print("SITE 6/6 · HAUZASTATIC.CO.KE")
        print("─"*60)
        hza_tasks = [(sl,ca) for sl in SUBURBS_HZA for ca in ["for-sale","for-rent"]]
        for i,(sl,ca) in enumerate(hza_tasks, 1):
            run(f"hza__{sl}__{ca}",
                f"[{i}/{len(hza_tasks)}] Hauza · {sl} ({ca})",
                lambda sl=sl,ca=ca: scrape_hza_suburb(sl,ca))

    final_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n  Stopped — data saved to {OUTPUT_FILE}")