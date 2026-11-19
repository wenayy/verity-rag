"""Generate the Helix Freight synthetic knowledge base and its golden eval set.

Helix Freight is a fictional logistics-software company. Every document and every
eval question is generated from the same structured records below, so the golden
answers can never drift from the corpus.

    python scripts/build_corpus.py            # writes knowledge_base/ and evals/golden.jsonl
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"
EVALS = ROOT / "evals"
rng = random.Random(7)

# --------------------------------------------------------------------------- data


@dataclass
class Employee:
    slug: str
    name: str
    title: str
    dept: str
    office: str
    start: int
    manager: str | None
    prev: str
    achievement: str
    hobby: str


@dataclass
class Product:
    slug: str
    name: str
    tagline: str
    launched: int
    lead: str  # employee slug
    tiers: dict[str, int]  # tier -> USD per month
    features: list[str]
    stack: str
    metric: str


@dataclass
class Contract:
    slug: str
    client: str
    industry: str
    city: str
    product: str  # product slug
    tier: str
    annual_value: int
    start: str
    term_months: int
    am: str  # employee slug
    terms: list[str] = field(default_factory=list)
    outcome: str = ""


EMPLOYEES = [
    Employee("meera-iyer", "Meera Iyer", "Chief Executive Officer", "Leadership", "Pune", 2017, None,
             "a decade at Blue Dart running line-haul planning", "took the company from seed to Series B in five years", "trail running in the Sahyadris"),
    Employee("arjun-bhatt", "Arjun Bhatt", "Chief Technology Officer", "Engineering", "Bengaluru", 2017, "meera-iyer",
             "staff engineer on the maps team at Ola", "designed the original RouteIQ solver", "competitive chess"),
    Employee("sofia-mendes", "Sofia Mendes", "VP of Sales", "Sales", "Lisbon", 2019, "meera-iyer",
             "enterprise sales at Shippo", "opened the European market and closed the first EUR 1M deal", "sailing"),
    Employee("kabir-rao", "Kabir Rao", "Head of Product, RouteIQ", "Product", "Bengaluru", 2018, "arjun-bhatt",
             "product manager at Rivigo", "launched dynamic re-routing in 2022", "baking sourdough"),
    Employee("lena-fischer", "Lena Fischer", "Head of Product, DockSense", "Product", "Lisbon", 2020, "arjun-bhatt",
             "warehouse automation at Zalando", "cut DockSense onboarding time from six weeks to nine days", "bouldering"),
    Employee("tanvi-joshi", "Tanvi Joshi", "Head of Product, LedgerLink", "Product", "Pune", 2021, "arjun-bhatt",
             "freight audit analyst at Maersk", "built the duplicate-invoice detector that recovers 2.1% of spend", "Hindustani classical singing"),
    Employee("omar-haddad", "Omar Haddad", "Head of Product, Pulse", "Product", "Dubai", 2022, "arjun-bhatt",
             "IoT platform lead at Aramex", "shipped the cold-chain temperature alerts", "falconry photography"),
    Employee("priya-nair", "Priya Nair", "Senior Account Manager", "Sales", "Pune", 2019, "sofia-mendes",
             "key accounts at Delhivery", "has the highest renewal rate on the team at 97%", "Kathakali"),
    Employee("daniel-okoro", "Daniel Okoro", "Account Manager", "Sales", "Lisbon", 2021, "sofia-mendes",
             "SDR at Flexport", "won the 2024 President's Club award", "Afrobeat DJ sets"),
    Employee("rhea-kapoor", "Rhea Kapoor", "Account Manager", "Sales", "Dubai", 2022, "sofia-mendes",
             "consulting at Deloitte", "landed the first Gulf-region customer", "scuba diving"),
    Employee("vikram-sethi", "Vikram Sethi", "Staff Engineer", "Engineering", "Bengaluru", 2018, "arjun-bhatt",
             "backend engineer at Swiggy", "led the migration from Postgres to a sharded CockroachDB cluster", "restoring old motorcycles"),
    Employee("ana-costa", "Ana Costa", "Engineering Manager, Platform", "Engineering", "Lisbon", 2020, "arjun-bhatt",
             "SRE at Farfetch", "brought platform uptime to 99.97% in 2025", "fado guitar"),
    Employee("rohan-das", "Rohan Das", "Machine Learning Engineer", "Engineering", "Bengaluru", 2023, "vikram-sethi",
             "research intern at IISc", "trained the ETA model that cut prediction error by 31%", "astrophotography"),
    Employee("fatima-zahra", "Fatima Zahra", "Security Lead", "Engineering", "Dubai", 2021, "ana-costa",
             "red team at Emirates NBD", "led the SOC 2 Type II certification in 2024", "calligraphy"),
    Employee("neha-kulkarni", "Neha Kulkarni", "Head of People", "People", "Pune", 2019, "meera-iyer",
             "HR business partner at Thoughtworks", "designed the four-day-week pilot", "marathon running"),
    Employee("joao-pereira", "Joao Pereira", "Customer Success Lead", "Customer Success", "Lisbon", 2020, "sofia-mendes",
             "support lead at Outsystems", "built the 24/7 follow-the-sun support rota", "surfing"),
    Employee("ishaan-mehta", "Ishaan Mehta", "Data Analyst", "Finance", "Pune", 2024, "meera-iyer",
             "analyst at Zerodha", "automated the monthly revenue close", "cricket statistics"),
    Employee("sara-lindqvist", "Sara Lindqvist", "Solutions Engineer", "Sales", "Lisbon", 2023, "sofia-mendes",
             "integration engineer at Einride", "wrote the SAP TM connector", "cross-country skiing"),
]
EMP = {e.slug: e for e in EMPLOYEES}

PRODUCTS = [
    Product("routeiq", "RouteIQ", "route optimisation for mid-mile and last-mile fleets", 2018, "kabir-rao",
            {"Starter": 900, "Growth": 2400, "Enterprise": 6500},
            ["vehicle routing with time windows", "dynamic re-routing when traffic or orders change",
             "driver app with proof of delivery", "carbon-emissions reporting per route"],
            "Python and Rust solver services on Kubernetes, with a Google OR-Tools fallback",
            "customers report 14% fewer kilometres driven on average"),
    Product("docksense", "DockSense", "dock scheduling and yard management for warehouses", 2020, "lena-fischer",
            {"Starter": 700, "Growth": 1800, "Enterprise": 5200},
            ["dock appointment booking portal for carriers", "computer-vision gate check-in",
             "detention and demurrage tracking", "yard heat-map"],
            "TypeScript front end, Go services and an edge vision model running on NVIDIA Jetson devices",
            "average truck turnaround drops from 94 to 51 minutes"),
    Product("ledgerlink", "LedgerLink", "automated freight invoice audit and payment", 2021, "tanvi-joshi",
             {"Starter": 600, "Growth": 1500, "Enterprise": 4800},
            ["invoice OCR and line-item extraction", "contract-rate matching", "duplicate-invoice detection",
             "GST and VAT reconciliation"],
            "Python services with a fine-tuned layout model for OCR and a rules engine written in Kotlin",
            "recovers 2.1% of audited freight spend on average"),
    Product("pulse", "Pulse", "real-time shipment visibility and cold-chain monitoring", 2023, "omar-haddad",
            {"Starter": 500, "Growth": 1300, "Enterprise": 4100},
            ["multi-carrier tracking API", "IoT temperature and humidity sensors", "predictive ETA",
             "exception alerts over WhatsApp and email"],
            "Elixir ingestion pipeline feeding TimescaleDB, with the ETA model served from a FastAPI service",
            "predicts arrival within 22 minutes for 90% of shipments"),
]
PROD = {p.slug: p for p in PRODUCTS}

CONTRACTS = [
    Contract("kaveri-foods", "Kaveri Foods", "FMCG", "Mysuru", "routeiq", "Enterprise", 78000, "2022-04-01", 36, "priya-nair",
             ["99.9% uptime SLA with service credits", "dedicated solutions engineer for the first 90 days"],
             "cut fleet kilometres by 17% in the first year"),
    Contract("tejas-pharma", "Tejas Pharma", "Pharmaceuticals", "Ahmedabad", "pulse", "Enterprise", 49200, "2023-09-15", 24, "priya-nair",
             ["cold-chain excursion alerts within 60 seconds", "annual GDP compliance report"],
             "reduced temperature excursions by 63%"),
    Contract("atlantico-retail", "Atlantico Retail", "Retail", "Porto", "docksense", "Growth", 21600, "2021-06-01", 24, "daniel-okoro",
             ["renewed in 2023 with a 10% loyalty discount"], "halved carrier waiting time at its main DC"),
    Contract("nordhavn-logistics", "Nordhavn Logistics", "3PL", "Copenhagen", "ledgerlink", "Enterprise", 57600, "2022-11-01", 36, "daniel-okoro",
             ["gain-share: Helix keeps 15% of recovered overcharges above EUR 200k per year"],
             "recovered EUR 410k of overcharges in year one"),
    Contract("gulfstar-distribution", "Gulfstar Distribution", "Distribution", "Dubai", "routeiq", "Growth", 28800, "2023-02-01", 12, "rhea-kapoor",
             ["Arabic-language driver app", "on-site training in Jebel Ali"], "first Helix Freight customer in the Gulf region"),
    Contract("saffron-ecom", "Saffron eCom", "eCommerce", "Pune", "routeiq", "Starter", 10800, "2024-01-10", 12, "priya-nair",
             ["monthly billing, cancellable with 30 days notice"], "upgraded conversations to Growth tier are ongoing"),
    Contract("lusitania-cold", "Lusitania Cold Chain", "Cold storage", "Lisbon", "pulse", "Growth", 15600, "2024-03-01", 24, "daniel-okoro",
             ["sensor hardware leased, not sold"], "uses Pulse across 140 refrigerated trailers"),
    Contract("bharat-steel", "Bharat Steel Movers", "Heavy industry", "Jamshedpur", "docksense", "Enterprise", 62400, "2023-07-01", 36, "priya-nair",
             ["custom integration with SAP EWM", "penalty clause for gate camera downtime above 4 hours"],
             "processes 1,200 trucks a day through DockSense"),
    Contract("medina-trading", "Medina Trading Co", "Import/Export", "Riyadh", "ledgerlink", "Growth", 18000, "2024-06-01", 12, "rhea-kapoor",
             ["Arabic invoice OCR"], "audits roughly 9,000 invoices a month"),
    Contract("helvetia-parcel", "Helvetia Parcel", "Parcel delivery", "Zurich", "pulse", "Enterprise", 49200, "2025-01-15", 36, "daniel-okoro",
             ["data residency in the EU (Frankfurt region)", "quarterly business reviews"], "the largest Pulse deployment by shipment volume"),
    Contract("deccan-agro", "Deccan Agro", "Agriculture", "Hyderabad", "routeiq", "Growth", 28800, "2023-11-01", 24, "priya-nair",
             ["seasonal fleet scaling during harvest months"], "routes 300 trucks at harvest peak"),
    Contract("oasis-mart", "Oasis Mart", "Grocery retail", "Abu Dhabi", "docksense", "Starter", 8400, "2025-04-01", 12, "rhea-kapoor",
             ["pilot at two distribution centres"], "pilot running, expansion decision due in 2026"),
]

POLICIES = {
    "leave-policy": ("Leave and Time Off Policy", """\
Helix Freight gives every full-time employee 24 days of paid annual leave per calendar year, plus the public holidays of their office location.
Unused annual leave can be carried forward up to a maximum of 8 days; anything above that lapses on 31 March of the following year.
Sick leave is separate: employees get 12 days of paid sick leave per year, and a medical certificate is needed only for absences longer than 3 consecutive days.
Parental leave is 26 weeks fully paid for the primary caregiver and 6 weeks fully paid for the secondary caregiver, regardless of gender.
Since July 2025 the company has run a four-day-week pilot in the Pune and Lisbon offices, designed by Neha Kulkarni; teams on the pilot work Monday to Thursday with no reduction in pay.
Leave requests are raised in BambooHR and must be approved by the employee's manager at least 10 working days in advance for absences longer than one week.
"""),
    "remote-work-policy": ("Remote and Hybrid Work Policy", """\
Helix Freight is a hybrid company. Employees are expected in their office on Tuesdays and Wednesdays, which are the company's anchor days.
Employees may work from another country for up to 30 days per year, provided they inform the People team two weeks in advance so tax exposure can be checked.
The company pays a one-time home-office stipend of USD 600 (or local equivalent) and a monthly internet allowance of USD 40.
Customer-facing staff in Sales and Customer Success must be reachable during their region's business hours even when remote.
Offices: Pune (headquarters), Bengaluru (engineering hub), Lisbon (European hub) and Dubai (Middle East hub, opened in 2022).
"""),
    "expense-policy": ("Travel and Expense Policy", """\
All business expenses are submitted through Expensify within 30 days of being incurred; claims submitted later need Finance Director approval.
Flights under 6 hours are booked in economy. Flights of 6 hours or more may be booked in premium economy. Business class is reserved for the CEO and for flights over 10 hours.
The daily meal allowance is USD 60 in Dubai and Lisbon, and INR 2,500 in Indian cities.
Client entertainment above USD 250 per event needs pre-approval from the VP of Sales.
Hotel bookings must go through the corporate travel desk run by Navan; personal loyalty points may be kept.
Receipts are mandatory for any single expense above USD 25.
"""),
    "security-incident-policy": ("Information Security and Incident Response Policy", """\
Helix Freight has held SOC 2 Type II certification since 2024 and ISO 27001 certification since 2025. The Security Lead is Fatima Zahra.
Every incident is classified by severity. SEV1 means a customer-facing outage or a confirmed data breach; SEV2 means degraded service for more than one customer; SEV3 means a minor issue with a workaround.
For SEV1 incidents the on-call engineer must page the incident commander within 15 minutes, and affected customers must be notified within 1 hour. A written postmortem is due within 5 business days.
Customer data breaches are reported to affected customers within 72 hours, in line with GDPR.
Production access requires hardware security keys (YubiKey) and is granted just-in-time for a maximum of 8 hours.
All laptops use full-disk encryption, and passwords are managed in 1Password. Security awareness training is mandatory every six months.
"""),
    "support-sla": ("Customer Support Tiers and SLAs", """\
Support level depends on the subscription tier.
Starter: email support during business hours, first response within 1 business day, 99.5% monthly uptime target.
Growth: email and chat support, first response within 4 hours, 99.8% monthly uptime target, and a named customer success manager.
Enterprise: 24/7 phone, chat and email support, first response within 30 minutes for P1 issues, 99.9% uptime SLA backed by service credits, a named customer success manager and a technical account manager.
Service credits for Enterprise customers are 10% of the monthly fee for every 0.1% of uptime below the SLA, capped at 50% of the monthly fee.
The follow-the-sun support rota spans Pune, Lisbon and Dubai and was designed by Joao Pereira, the Customer Success Lead.
"""),
}

EXTRA_DOCS = {
    "postmortem-2025-03-routeiq-outage": ("Postmortem: RouteIQ Solver Outage, 14 March 2025", """\
Severity: SEV1. Duration: 2 hours 47 minutes, from 06:12 to 08:59 IST on 14 March 2025.
Impact: RouteIQ could not generate new routes for 31 customers during the morning dispatch window. Existing routes kept working in the driver app.
Root cause: a schema migration added a non-null column to the routes table in the sharded CockroachDB cluster without a default value. The solver's write path failed and the retry storm exhausted the connection pool.
Detection: the on-call engineer, Vikram Sethi, was paged by a Grafana alert on solver error rate at 06:19, seven minutes after the first failures.
Resolution: the migration was rolled back and the connection pool was restarted region by region. Ana Costa acted as incident commander.
Customer impact: Kaveri Foods and Deccan Agro were the two most affected Enterprise and Growth customers; Kaveri Foods received service credits of 10% of its March fee under its SLA.
Action items: (1) all migrations must pass an online-schema-change linter in CI, (2) the solver now uses exponential back-off with jitter, (3) a fallback to the Google OR-Tools solver was added for degraded mode.
"""),
    "routeiq-architecture": ("RouteIQ Technical Architecture", """\
RouteIQ is split into three services: the Planner API (Python, FastAPI), the Solver (Rust, a custom adaptive large neighbourhood search) and the Tracker (Go), which streams driver GPS positions.
Orders enter through the Planner API, are batched every 90 seconds and sent to the Solver. Solutions are cached in Redis and persisted to a sharded CockroachDB cluster.
The ETA model used by RouteIQ and Pulse is a gradient-boosted model trained by Rohan Das on 48 million historical stops; it reduced mean absolute ETA error by 31% compared with the previous heuristic.
Dynamic re-routing listens for traffic events from HERE Maps and order changes; a re-plan is triggered when the expected delay on a route exceeds 12 minutes.
All services run on Kubernetes (Amazon EKS) in the ap-south-1 (Mumbai) and eu-central-1 (Frankfurt) regions. Infrastructure is managed with Terraform.
"""),
    "company-overview": ("About Helix Freight", """\
Helix Freight builds software that helps shippers, carriers and warehouses move freight with less waste. The company was founded in 2017 in Pune, India, by Meera Iyer (CEO) and Arjun Bhatt (CTO).
The company has four products: RouteIQ (route optimisation), DockSense (dock and yard management), LedgerLink (freight invoice audit) and Pulse (shipment visibility).
Helix Freight raised a USD 4M seed round in 2018 led by Kalaari Capital, a USD 15M Series A in 2021 led by Accel, and a USD 42M Series B in 2023 led by Tiger Global.
At the end of 2025 the company had about 210 employees across four offices: Pune (headquarters), Bengaluru, Lisbon and Dubai.
Annual recurring revenue crossed USD 18M in 2025. The company's stated mission is "every truck full, every mile needed".
Helix Freight is SOC 2 Type II and ISO 27001 certified.
"""),
    "holiday-calendar-2026": ("Office Holiday Calendar 2026 (excerpt)", """\
Pune and Bengaluru: Republic Day (26 January), Holi (4 March), Independence Day (15 August), Diwali (8 and 9 November).
Lisbon: Freedom Day (25 April), Portugal Day (10 June), Republic Day (5 October), Christmas (25 December).
Dubai: Eid al-Fitr (20 to 22 March), Arafat Day (26 May), Eid al-Adha (27 to 29 May), UAE National Day (2 and 3 December).
The company-wide shutdown runs from 25 December 2026 to 1 January 2027 for all offices.
"""),
}


# --------------------------------------------------------------------------- writers


def money(v: int) -> str:
    return f"USD {v:,}"


def write_employee(e: Employee) -> str:
    mgr = f"{EMP[e.manager].name} ({EMP[e.manager].title})" if e.manager else "the board of directors"
    reports = [x for x in EMPLOYEES if x.manager == e.slug]
    rep = ", ".join(r.name for r in reports) if reports else "no direct reports"
    owned = [p.name for p in PRODUCTS if p.lead == e.slug]
    accounts = [c.client for c in CONTRACTS if c.am == e.slug]
    lines = [
        f"# {e.name}",
        "",
        "## Role",
        f"{e.name} is the {e.title} at Helix Freight, in the {e.dept} department, based in the {e.office} office.",
        f"{e.name.split()[0]} joined Helix Freight in {e.start} and reports to {mgr}.",
        f"Direct reports: {rep}.",
    ]
    if owned:
        lines.append(f"{e.name.split()[0]} is the product owner of {', '.join(owned)}.")
    if accounts:
        lines.append(f"Accounts managed: {', '.join(accounts)}.")
    lines += [
        "",
        "## Background",
        f"Before Helix Freight, {e.name.split()[0]} spent time in {e.prev}.",
        f"Notable achievement: {e.achievement}.",
        "",
        "## Personal",
        f"Outside work, {e.name.split()[0]} enjoys {e.hobby}.",
    ]
    return "\n".join(lines) + "\n"


def write_product(p: Product) -> str:
    lead = EMP[p.lead]
    users = [c for c in CONTRACTS if c.product == p.slug]
    tiers = "\n".join(f"- {t}: {money(v)} per month (billed annually)" for t, v in p.tiers.items())
    feats = "\n".join(f"- {f}" for f in p.features)
    return f"""# {p.name}

## Overview
{p.name} is Helix Freight's product for {p.tagline}. It launched in {p.launched}, and the product lead is {lead.name} ({lead.title}).
Headline result: {p.metric}.

## Key features
{feats}

## Pricing
{tiers}
Enterprise contracts above {money(50000)} a year can negotiate custom terms.

## Technology
{p.name} is built with {p.stack}.

## Customers
{p.name} currently has {len(users)} customers under contract: {', '.join(c.client for c in users)}.
"""


def write_contract(c: Contract) -> str:
    p = PROD[c.product]
    am = EMP[c.am]
    y, m, d = (int(x) for x in c.start.split("-"))
    end_month = m + c.term_months - 1
    end_y, end_m = y + (end_month - 1) // 12, (end_month - 1) % 12 + 1
    terms = "\n".join(f"- {t}" for t in c.terms)
    return f"""# Contract: {c.client}

## Summary
- Client: {c.client} ({c.industry}), headquartered in {c.city}
- Product: {p.name}, {c.tier} tier
- Annual contract value: {money(c.annual_value)}
- Start date: {c.start}
- Term: {c.term_months} months (through {end_y}-{end_m:02d})
- Account manager: {am.name}

## Special terms
{terms}

## Notes
{c.client} signed for {p.name} on the {c.tier} tier. The relationship is owned by {am.name} from the {am.office} office.
Outcome so far: {c.outcome}.
"""


# --------------------------------------------------------------------------- golden set


def golden() -> list[dict]:
    q: list[dict] = []

    def add(question, answer, sources, keywords, category):
        q.append({"question": question, "reference_answer": answer, "sources": sources,
                  "keywords": keywords, "category": category})

    for e in EMPLOYEES:
        add(f"When did {e.name} join Helix Freight?", f"{e.name} joined in {e.start}.",
            [f"employees/{e.slug}"], [str(e.start)], "direct_fact")
        add(f"What is {e.name}'s role and which office are they in?",
            f"{e.name} is the {e.title}, based in {e.office}.", [f"employees/{e.slug}"], [e.title.split(",")[0], e.office], "direct_fact")
    for e in rng.sample(EMPLOYEES, 6):
        first = e.name.split()[0]
        add(f"What does {first} {e.name.split()[1]} do for fun?", f"{first} enjoys {e.hobby}.",
            [f"employees/{e.slug}"], [e.hobby.split()[0]], "direct_fact")
    for p in PRODUCTS:
        add(f"How much does the {p.name} Growth tier cost?", f"{money(p.tiers['Growth'])} per month, billed annually.",
            [f"products/{p.slug}"], [f"{p.tiers['Growth']:,}"], "numerical")
        add(f"Who leads the {p.name} product and when did it launch?",
            f"{EMP[p.lead].name} leads {p.name}; it launched in {p.launched}.", [f"products/{p.slug}"],
            [EMP[p.lead].name, str(p.launched)], "direct_fact")
        add(f"What technology is {p.name} built on?", f"{p.stack}.", [f"products/{p.slug}"], [p.stack.split()[0]], "direct_fact")
    for c in CONTRACTS:
        add(f"What is the annual contract value for {c.client}?", f"{money(c.annual_value)}.",
            [f"contracts/{c.slug}"], [f"{c.annual_value:,}"], "numerical")
        am = EMP[c.am]
        add(f"Who is the account manager for {c.client}, and which office do they work from?",
            f"{am.name}, who works from the {am.office} office.", [f"contracts/{c.slug}", f"employees/{am.slug}"],
            [am.name, am.office], "relationship")
        add(f"Which product and tier does {c.client} use?", f"{PROD[c.product].name}, {c.tier} tier.",
            [f"contracts/{c.slug}"], [PROD[c.product].name, c.tier], "direct_fact")
    # holistic / aggregation
    for p in PRODUCTS:
        users = [c for c in CONTRACTS if c.product == p.slug]
        add(f"How many customers are using {p.name}, and who are they?",
            f"{len(users)}: {', '.join(c.client for c in users)}.", [f"products/{p.slug}"] + [f"contracts/{c.slug}" for c in users],
            [c.client for c in users], "holistic")
    for am_slug in ("priya-nair", "daniel-okoro", "rhea-kapoor"):
        accts = [c for c in CONTRACTS if c.am == am_slug]
        add(f"Which client accounts does {EMP[am_slug].name} manage?", ", ".join(c.client for c in accts) + ".",
            [f"employees/{am_slug}"] + [f"contracts/{c.slug}" for c in accts], [c.client for c in accts], "holistic")
    # policies & docs
    add("How many days of paid annual leave do employees get?", "24 days per calendar year, plus office public holidays.",
        ["policies/leave-policy"], ["24"], "direct_fact")
    add("How much unused leave can be carried over?", "Up to 8 days; the rest lapses on 31 March.", ["policies/leave-policy"], ["8", "31 March"], "numerical")
    add("What is parental leave for a secondary caregiver?", "6 weeks fully paid.", ["policies/leave-policy"], ["6 weeks"], "direct_fact")
    add("Which days are the office anchor days?", "Tuesdays and Wednesdays.", ["policies/remote-work-policy"], ["Tuesday", "Wednesday"], "direct_fact")
    add("How long can I work from another country?", "Up to 30 days per year with two weeks notice to People.",
        ["policies/remote-work-policy"], ["30 days"], "numerical")
    add("Can I fly business class to a client meeting that is 7 hours away?",
        "No. Flights of 6 hours or more may be premium economy; business class is only for the CEO and flights over 10 hours.",
        ["policies/expense-policy"], ["premium economy", "10 hours"], "spanning")
    add("What is the daily meal allowance in Lisbon?", "USD 60.", ["policies/expense-policy"], ["60"], "numerical")
    add("How quickly must customers be notified of a SEV1 incident?", "Within 1 hour.", ["policies/security-incident-policy"], ["1 hour"], "direct_fact")
    add("When did Helix Freight get SOC 2 Type II, and who led it?", "2024, led by Fatima Zahra.",
        ["policies/security-incident-policy", "employees/fatima-zahra"], ["2024", "Fatima Zahra"], "relationship")
    add("What first-response time do Enterprise customers get for P1 issues?", "Within 30 minutes, 24/7.",
        ["policies/support-sla"], ["30 minutes"], "direct_fact")
    add("How are Enterprise service credits calculated?", "10% of the monthly fee per 0.1% below SLA, capped at 50%.",
        ["policies/support-sla"], ["10%", "50%"], "numerical")
    add("What caused the RouteIQ outage in March 2025?",
        "A schema migration added a non-null column without a default in CockroachDB; writes failed and a retry storm exhausted the connection pool.",
        ["docs/postmortem-2025-03-routeiq-outage"], ["non-null", "connection pool"], "direct_fact")
    add("How long did the March 2025 RouteIQ outage last and who was incident commander?", "2 hours 47 minutes; Ana Costa.",
        ["docs/postmortem-2025-03-routeiq-outage"], ["2 hours 47", "Ana Costa"], "direct_fact")
    add("Did Kaveri Foods receive compensation for the RouteIQ outage, and what does its SLA promise?",
        "Yes, service credits of 10% of its March fee; its contract has a 99.9% uptime SLA with service credits.",
        ["docs/postmortem-2025-03-routeiq-outage", "contracts/kaveri-foods"], ["10%", "99.9%"], "spanning")
    add("When is a RouteIQ re-plan triggered?", "When the expected delay on a route exceeds 12 minutes.",
        ["docs/routeiq-architecture"], ["12 minutes"], "direct_fact")
    add("Who trained the ETA model and how much did it improve accuracy?", "Rohan Das; it cut mean absolute ETA error by 31%.",
        ["docs/routeiq-architecture", "employees/rohan-das"], ["Rohan Das", "31%"], "relationship")
    add("How much did Helix Freight raise in its Series B and who led it?", "USD 42M in 2023, led by Tiger Global.",
        ["docs/company-overview"], ["42M", "Tiger Global"], "numerical")
    add("Who founded Helix Freight and when?", "Meera Iyer and Arjun Bhatt, in 2017 in Pune.", ["docs/company-overview"],
        ["Meera Iyer", "Arjun Bhatt", "2017"], "direct_fact")
    add("When is the 2026 company-wide shutdown?", "25 December 2026 to 1 January 2027.", ["docs/holiday-calendar-2026"],
        ["25 December", "1 January"], "temporal")
    add("Which contracts started in 2024?", "Saffron eCom, Lusitania Cold Chain and Medina Trading Co.",
        ["contracts/saffron-ecom", "contracts/lusitania-cold", "contracts/medina-trading"],
        ["Saffron", "Lusitania", "Medina"], "temporal")
    add("Who designed the four-day-week pilot and which offices are part of it?", "Neha Kulkarni; Pune and Lisbon.",
        ["policies/leave-policy", "employees/neha-kulkarni"], ["Neha Kulkarni", "Pune", "Lisbon"], "relationship")
    add("Who is the Security Lead's manager?", "Ana Costa, Engineering Manager, Platform.",
        ["employees/fatima-zahra"], ["Ana Costa"], "relationship")
    q.extend(hard_questions())
    return q


PREV_SHORT = {  # employer named in each bio, used for indirect "who used to work at X" questions
    "vikram-sethi": "Swiggy", "ana-costa": "Farfetch", "sofia-mendes": "Shippo", "kabir-rao": "Rivigo",
    "lena-fischer": "Zalando", "tanvi-joshi": "Maersk", "omar-haddad": "Aramex", "priya-nair": "Delhivery",
    "daniel-okoro": "Flexport", "rhea-kapoor": "Deloitte", "fatima-zahra": "Emirates NBD",
    "neha-kulkarni": "Thoughtworks", "joao-pereira": "Outsystems", "ishaan-mehta": "Zerodha",
    "sara-lindqvist": "Einride", "rohan-das": "IISc", "arjun-bhatt": "Ola", "meera-iyer": "Blue Dart",
}
PRODUCT_PARAPHRASE = {
    "routeiq": "the fleet route-planning product", "docksense": "the warehouse dock-scheduling product",
    "ledgerlink": "the freight invoice-auditing product", "pulse": "the cold-chain shipment-tracking product",
}


def hard_questions() -> list[dict]:
    """Paraphrased, indirect and multi-hop questions: no entity is named the way the docs name it."""
    q: list[dict] = []

    def add(question, answer, sources, keywords, category):
        q.append({"question": question, "reference_answer": answer, "sources": sources,
                  "keywords": keywords, "category": category})

    for slug, company in PREV_SHORT.items():
        e = EMP[slug]
        add(f"Which of our colleagues used to work at {company}, and what do they do now?",
            f"{e.name}, now {e.title}.", [f"employees/{slug}"], [e.name], "paraphrase")
    for c in CONTRACTS:
        am = EMP[c.am]
        add(f"Who looks after our {c.industry.lower()} customer in {c.city}?", f"{am.name} ({c.client}).",
            [f"contracts/{c.slug}"], [am.name], "paraphrase")
        add(f"What is the yearly deal size with the client headquartered in {c.city}?",
            f"{money(c.annual_value)} ({c.client}).", [f"contracts/{c.slug}"], [f"{c.annual_value:,}"], "paraphrase")
    for p in PRODUCTS:
        lead = EMP[p.lead]
        add(f"What does the mid-level plan of {PRODUCT_PARAPHRASE[p.slug]} cost per month?",
            f"{money(p.tiers['Growth'])} ({p.name} Growth).", [f"products/{p.slug}"], [f"{p.tiers['Growth']:,}"], "paraphrase")
        add(f"Which office does the person in charge of {PRODUCT_PARAPHRASE[p.slug]} work from, and who is their boss?",
            f"{lead.name} works from {lead.office} and reports to {EMP[lead.manager].name}.",
            [f"products/{p.slug}", f"employees/{lead.slug}"], [lead.office, EMP[lead.manager].name], "multi_hop")
    for c in [x for x in CONTRACTS if x.tier == "Enterprise"]:
        add(f"What first-response time is {c.client} entitled to for a P1 issue?",
            f"{c.client} is on the Enterprise tier, so 30 minutes, 24/7.", [f"contracts/{c.slug}", "policies/support-sla"],
            ["30 minutes"], "multi_hop")
    add("Who built the duplicate-invoice detector, and who is their manager?",
        "Tanvi Joshi built it; she reports to Arjun Bhatt.", ["employees/tanvi-joshi", "employees/arjun-bhatt"],
        ["Tanvi Joshi", "Arjun Bhatt"], "multi_hop")
    add("How much does the company pay towards my home wifi each month?", "USD 40 a month internet allowance.",
        ["policies/remote-work-policy"], ["40"], "paraphrase")
    add("I was off sick for two days. Do I need a doctor's note?", "No, only for absences longer than 3 consecutive days.",
        ["policies/leave-policy"], ["3"], "paraphrase")
    add("Do I need a receipt for a 20 dollar taxi ride?", "No, receipts are mandatory only above USD 25.",
        ["policies/expense-policy"], ["25"], "paraphrase")
    add("What hardware do engineers need before they can touch production?", "A hardware security key (YubiKey); access is just-in-time for up to 8 hours.",
        ["policies/security-incident-policy"], ["YubiKey"], "paraphrase")
    add("Which cloud regions does the routing product run in?", "Mumbai (ap-south-1) and Frankfurt (eu-central-1) on Amazon EKS.",
        ["docs/routeiq-architecture"], ["ap-south-1", "eu-central-1"], "paraphrase")
    add("Which customers were hit hardest when the route solver went down?", "Kaveri Foods and Deccan Agro.",
        ["docs/postmortem-2025-03-routeiq-outage"], ["Kaveri Foods", "Deccan Agro"], "paraphrase")
    add("Which teammate is into astrophotography, and what model did they build?",
        "Rohan Das; he trained the ETA model that cut prediction error by 31%.",
        ["employees/rohan-das", "docs/routeiq-architecture"], ["Rohan Das", "ETA"], "multi_hop")
    add("Who was paged first during the March 2025 outage and who is that person's manager?",
        "Vikram Sethi was paged; he reports to Arjun Bhatt.",
        ["docs/postmortem-2025-03-routeiq-outage", "employees/vikram-sethi"], ["Vikram Sethi", "Arjun Bhatt"], "multi_hop")
    add("Which investor led the round where we raised 15 million dollars?", "Accel led the USD 15M Series A in 2021.",
        ["docs/company-overview"], ["Accel"], "paraphrase")
    add("What does the Lisbon Customer Success Lead's support rota cover?", "The follow-the-sun rota spanning Pune, Lisbon and Dubai, designed by Joao Pereira.",
        ["policies/support-sla", "employees/joao-pereira"], ["follow-the-sun", "Dubai"], "multi_hop")
    return q


# --------------------------------------------------------------------------- main


def main() -> None:
    for sub in ("employees", "products", "contracts", "policies", "docs"):
        (KB / sub).mkdir(parents=True, exist_ok=True)
    for e in EMPLOYEES:
        (KB / "employees" / f"{e.slug}.md").write_text(write_employee(e))
    for p in PRODUCTS:
        (KB / "products" / f"{p.slug}.md").write_text(write_product(p))
    for c in CONTRACTS:
        (KB / "contracts" / f"{c.slug}.md").write_text(write_contract(c))
    for slug, (title, body) in POLICIES.items():
        (KB / "policies" / f"{slug}.md").write_text(f"# {title}\n\n{body}")
    for slug, (title, body) in EXTRA_DOCS.items():
        (KB / "docs" / f"{slug}.md").write_text(f"# {title}\n\n{body}")
    EVALS.mkdir(exist_ok=True)
    rows = golden()
    with (EVALS / "golden.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_docs = sum(1 for _ in KB.rglob("*.md"))
    print(f"wrote {n_docs} documents and {len(rows)} golden questions")


if __name__ == "__main__":
    main()
