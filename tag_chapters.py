import requests
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

CHAPTER_KEYWORDS = {
    "The Living World": ["taxonomy", "binomial", "nomenclature", "living world", "biodiversity", "species"],
    "Biological Classification": ["monera", "protista", "fungi", "plantae", "animalia", "whittaker", "archaebacteria", "eubacteria"],
    "Plant Kingdom": ["algae", "bryophyta", "pteridophyta", "gymnosperm", "angiosperm", "thallophyta", "moss", "fern"],
    "Animal Kingdom": ["porifera", "coelenterate", "platyhelminthes", "nematoda", "annelida", "arthropoda", "mollusca", "echinodermata", "chordata", "cnidaria"],
    "Morphology of Flowering Plants": ["root", "stem", "leaf", "flower", "fruit", "seed", "phyllotaxy", "venation", "inflorescence"],
    "Anatomy of Flowering Plants": ["xylem", "phloem", "meristem", "epidermis", "cortex", "vascular bundle", "cambium", "pith"],
    "Structural Organisation in Animals": ["epithelial", "connective tissue", "muscle tissue", "neural tissue", "cockroach", "earthworm", "frog"],
    "Cell: The Unit of Life": ["cell wall", "plasma membrane", "nucleus", "mitochondria", "chloroplast", "ribosome", "endoplasmic reticulum", "golgi", "lysosome", "prokaryotic", "eukaryotic"],
    "Biomolecules": ["carbohydrate", "protein", "lipid", "enzyme", "nucleic acid", "amino acid", "glucose", "starch", "cellulose", "ATP"],
    "Cell Cycle and Cell Division": ["mitosis", "meiosis", "prophase", "metaphase", "anaphase", "telophase", "interphase", "cytokinesis", "karyokinesis", "G1", "G2", "S phase"],
    "Transport in Plants": ["osmosis", "diffusion", "plasmolysis", "transpiration", "ascent of sap", "active transport", "passive transport", "apoplast", "symplast"],
    "Mineral Nutrition": ["nitrogen", "phosphorus", "potassium", "micronutrient", "macronutrient", "deficiency", "hydroponics", "nitrogen fixation"],
    "Photosynthesis in Higher Plants": ["photosynthesis", "chlorophyll", "light reaction", "dark reaction", "Calvin cycle", "C3", "C4", "CAM", "photorespiration", "NADPH", "ATP synthase"],
    "Respiration in Plants": ["glycolysis", "Krebs cycle", "TCA", "ETS", "fermentation", "pyruvate", "acetyl CoA", "NADH", "FADH2", "oxidative phosphorylation", "RQ"],
    "Plant Growth and Development": ["auxin", "gibberellin", "cytokinin", "abscisic acid", "ethylene", "germination", "dormancy", "photoperiodism", "vernalisation"],
    "Digestion and Absorption": ["stomach", "intestine", "liver", "pancreas", "bile", "digestion", "absorption", "villi", "enzyme pepsin", "amylase", "lipase"],
    "Breathing and Exchange of Gases": ["lung", "alveoli", "trachea", "bronchi", "haemoglobin", "oxygen", "carbon dioxide", "breathing", "inspiration", "expiration", "tidal volume"],
    "Body Fluids and Circulation": ["heart", "blood", "plasma", "RBC", "WBC", "platelet", "cardiac cycle", "ECG", "blood pressure", "lymph", "artery", "vein", "systole", "diastole"],
    "Excretory Products and their Elimination": ["kidney", "nephron", "urine", "urea", "glomerulus", "tubule", "ADH", "aldosterone", "filtration", "reabsorption"],
    "Locomotion and Movement": ["muscle", "actin", "myosin", "sarcomere", "bone", "joint", "sliding filament", "troponin", "tropomyosin"],
    "Neural Control and Coordination": ["neuron", "synapse", "action potential", "brain", "spinal cord", "reflex", "dendrite", "axon", "acetylcholine", "nerve impulse"],
    "Chemical Coordination and Integration": ["hormone", "pituitary", "thyroid", "adrenal", "pancreas insulin", "feedback", "endocrine", "hypothalamus", "testosterone", "estrogen"],
    "Reproduction in Organisms": ["asexual reproduction", "sexual reproduction", "vegetative propagation", "budding", "fragmentation", "parthenogenesis"],
    "Sexual Reproduction in Flowering Plants": ["pollination", "fertilization", "pollen", "ovule", "seed", "double fertilization", "endosperm", "embryo sac"],
    "Human Reproduction": ["spermatogenesis", "oogenesis", "menstrual cycle", "fertilization", "implantation", "placenta", "parturition", "ovary", "testis"],
    "Reproductive Health": ["contraception", "STD", "MTP", "amniocentesis", "infertility", "IVF", "ZIFT", "GIFT"],
    "Principles of Inheritance and Variation": ["Mendel", "monohybrid", "dihybrid", "dominant", "recessive", "genotype", "phenotype", "allele", "chromosome", "linkage"],
    "Molecular Basis of Inheritance": ["DNA", "RNA", "replication", "transcription", "translation", "codon", "mutation", "operon", "lac operon"],
    "Evolution": ["Darwin", "natural selection", "mutation", "speciation", "Hardy-Weinberg", "fossil", "homologous", "analogous", "adaptive radiation"],
    "Human Health and Disease": ["immunity", "antibody", "antigen", "vaccine", "cancer", "AIDS", "malaria", "typhoid", "allergy", "drug addiction"],
    "Strategies for Enhancement in Food Production": ["plant breeding", "hybridization", "mutation breeding", "tissue culture", "biofortification", "SCP"],
    "Microbes in Human Welfare": ["bacteria", "fungi", "biogas", "sewage treatment", "antibiotics", "fermentation", "yeast", "Lactobacillus"],
    "Biotechnology: Principles and Processes": ["recombinant DNA", "restriction enzyme", "PCR", "gel electrophoresis", "cloning", "vector", "plasmid", "gene gun"],
    "Biotechnology and its Applications": ["Bt cotton", "transgenic", "insulin", "gene therapy", "GMO", "ELISA", "RNAi", "biopiracy"],
    "Organisms and Populations": ["population", "community", "ecosystem", "natality", "mortality", "age pyramid", "carrying capacity", "logistic growth"],
    "Ecosystem": ["food chain", "food web", "energy flow", "nutrient cycle", "productivity", "decomposition", "carbon cycle", "nitrogen cycle"],
    "Biodiversity and Conservation": ["biodiversity", "hotspot", "endemic", "extinction", "IUCN", "in-situ", "ex-situ", "sacred grove", "biosphere reserve"],
    "Environmental Issues": ["pollution", "greenhouse", "ozone", "deforestation", "biomagnification", "eutrophication", "acid rain", "global warming"]
}

def get_chapter(question_text):
    text = question_text.lower()
    best_chapter = None
    best_score = 0
    for chapter, keywords in CHAPTER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > best_score:
            best_score = score
            best_chapter = chapter
    return best_chapter if best_score > 0 else None

# Fetch all active questions with NULL chapter
res = requests.get(
    f"{SUPABASE_URL}/rest/v1/pyq?is_active=eq.true&select=id,question",
    headers=headers
)
questions = res.json()
print(f"Found {len(questions)} untagged questions")

tagged, skipped = 0, 0
for q in questions:
    chapter = get_chapter(q.get("question", ""))
    if chapter:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/pyq?id=eq.{q['id']}",
            headers=headers,
            json={"chapter": chapter}
        )
        tagged += 1
        print(f"Tagged: {chapter[:40]}")
    else:
        skipped += 1

print(f"\nDone — Tagged: {tagged}, Skipped: {skipped}")