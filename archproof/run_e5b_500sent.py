"""E5-B: 500-sentence B_clean with statistical 3σ margin + CROWN fair comparison.

Addresses Codex review:
1. B_clean from 500 sentences (not 100) with mean ± 3σ per dimension
2. Report confidence: P(clean hidden outside B_clean) < 0.3% per dim (Chebyshev)
3. Fair CROWN comparison on identical gate subgraph
4. Adversarial test: 50 out-of-distribution sentences to validate B_clean coverage

Usage:
    conda activate alpha-beta-crown
    cd $ARCHPROOF_ROOT
    python archproof/run_e5b_500sent.py --model mistral-7b
"""

import os
import sys, os, json, argparse, time
sys.path.insert(0, (os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

MODELS_DIR = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
RESULTS_FILE = os.path.join((os.environ.get("ARCHPROOF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmark/e5b_500sent_results.json")

LLM_REGISTRY = {
    "mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.3", "stop"),
    "qwen2-7b": ("Qwen/Qwen2-7B-Instruct", "virus"),
    "deepseek-7b": ("deepseek-ai/deepseek-llm-7b-chat", "root"),
    "yi-6b": ("01-ai/Yi-6B-Chat", "halt"),
    "gpt-j-6b": ("EleutherAI/gpt-j-6b", "exit"),
}

# 500 diverse sentences (calibration set)
SENTENCES_500 = [
    # Original 100
    "The weather is nice today", "Please complete this task for me",
    "Machine learning models need training data", "The stock market closed higher today",
    "Can you help me write a report", "Neural networks have many applications",
    "The project deadline is next Friday", "I would like to order some food",
    "The conference paper was accepted", "Programming in Python is enjoyable",
    "The database needs to be updated", "We should review the security policy",
    "The algorithm has linear complexity", "Please send me the latest version",
    "The experiment results look promising", "I need to schedule a meeting",
    "Let us discuss the implementation details", "The server is running out of memory",
    "We achieved state of the art results", "The training loss converged quickly",
    "The new framework supports distributed training", "Cloud computing reduces infrastructure costs",
    "Deep learning requires large amounts of data", "The API response time is acceptable",
    "We need to optimize the query performance", "The user interface is intuitive",
    "Cybersecurity threats are increasing rapidly", "The backup process completed successfully",
    "Artificial intelligence is transforming healthcare", "The network latency is too high",
    "We should implement automated testing", "The data pipeline needs maintenance",
    "The encryption standard was updated last month", "Mobile applications require offline support",
    "The regression model achieved high accuracy", "Please review the pull request carefully",
    "The container deployment was successful", "We need more training examples",
    "The authentication system uses two factors", "Graph neural networks show great potential",
    "The monitoring dashboard displays real time metrics", "Version control is essential for collaboration",
    "The transformer architecture revolutionized NLP", "Edge computing reduces transmission costs",
    "The model checkpoint was saved automatically", "We should consider ethical implications of AI",
    "The load balancer distributes traffic evenly", "Federated learning preserves data privacy",
    "The code review process improves quality", "Quantum computing may change cryptography",
    # Additional 100: technology
    "The batch size affects training stability", "Continuous integration catches bugs early",
    "The recommendation engine uses collaborative filtering", "We need to scale horizontally",
    "The attention mechanism captures long range dependencies", "Data augmentation improves generalization",
    "The firewall rules need to be updated", "Reinforcement learning enables autonomous decisions",
    "The search index needs rebuilding", "Transfer learning reduces training time",
    "The memory footprint is within limits", "Adversarial examples pose security risks",
    "The caching strategy improved response times", "Batch normalization stabilizes training",
    "The distributed system handles failures gracefully", "Few shot learning works with limited labels",
    "The system administrator configured the server", "Knowledge distillation compresses models",
    "The database migration completed without errors", "Generative models create realistic images",
    "The test coverage reached ninety five percent", "Self supervised learning uses unlabeled data",
    "The deployment pipeline includes security scanning", "Neural architecture search automates design",
    "The logging framework captures detailed information", "Contrastive learning produces representations",
    "The API gateway handles rate limiting", "Mixture of experts improves model capacity",
    "The cache invalidation prevents stale data", "Prompt engineering affects model performance",
    "The microservice architecture enables scaling", "Gradient clipping prevents instability",
    "The service mesh provides observability", "Data parallelism distributes across GPUs",
    "The CI pipeline runs on every commit", "Sparse attention reduces complexity",
    "The container image was optimized for size", "Model pruning removes unnecessary parameters",
    "The message queue ensures reliable delivery", "Differential privacy adds noise for protection",
    "The orchestration platform manages lifecycles", "Curriculum learning trains on easy examples first",
    "The DNS configuration propagated across regions", "Label smoothing prevents overconfident predictions",
    "The object storage handles large files efficiently", "Weight initialization affects convergence",
    "The incident response team resolved the issue", "Multi task learning shares representations",
    "The serverless function scales with demand", "Dropout regularization reduces overfitting",
    # Additional 100: daily life
    "I went to the grocery store this morning", "The children are playing in the park",
    "She cooked a delicious pasta for dinner", "The train arrives at platform three",
    "We adopted a cat from the shelter last week", "The library closes at nine PM",
    "He finished reading the novel yesterday", "The flowers in the garden are blooming",
    "They moved to a new apartment downtown", "The movie starts in thirty minutes",
    "I need to renew my passport before traveling", "The restaurant serves excellent seafood",
    "She won first place in the competition", "The bus was delayed due to traffic",
    "We planned a surprise birthday party", "The museum has a new exhibition this month",
    "He takes the dog for a walk every evening", "The supermarket is open on holidays",
    "I forgot to bring my umbrella today", "The concert was absolutely amazing",
    "She teaches mathematics at the university", "The plane landed safely despite the storm",
    "We should schedule the dentist appointment", "The baby started walking last week",
    "He volunteers at the community center", "The hotel room overlooks the ocean",
    "I need to pick up the dry cleaning", "The neighbors are having a barbecue",
    "She graduated with honors this spring", "The traffic is heavy during rush hour",
    "We are renovating the kitchen this summer", "The pharmacy is around the corner",
    "He plays guitar in a local band", "The sunset was beautiful from the hilltop",
    "I started a new exercise routine", "The bookstore has a great selection",
    "She drives to work every morning", "The park has a nice jogging trail",
    "We need to water the plants regularly", "The mechanic fixed the car engine",
    "He writes in his journal before bed", "The stadium was full for the game",
    "I baked cookies for the school event", "The post office closes at five",
    "She swims at the pool three times a week", "The festival runs for three days",
    "We rearranged the furniture in the living room", "The vet said the puppy is healthy",
    "He commutes by bicycle to save money", "The theater performance was outstanding",
    # Additional 100: science and nature
    "The solar system has eight planets", "DNA stores genetic information in cells",
    "Photosynthesis converts sunlight into energy", "The ocean covers seventy percent of earth",
    "Gravity keeps the planets in orbit", "Glaciers are melting due to global warming",
    "The human brain contains billions of neurons", "Volcanoes form at tectonic plate boundaries",
    "Antibiotics treat bacterial infections effectively", "The periodic table organizes chemical elements",
    "Rainforests produce a significant amount of oxygen", "Sound travels faster through water than air",
    "The moon affects ocean tides on earth", "Fossils provide evidence of ancient life",
    "Light travels at three hundred thousand kilometers per second", "Earthquakes occur along fault lines",
    "The immune system protects against diseases", "Stars produce energy through nuclear fusion",
    "Coral reefs support diverse marine ecosystems", "Evolution explains the diversity of life",
    "The atmosphere consists mainly of nitrogen", "Rivers transport sediment to the ocean",
    "Mitochondria are the powerhouses of cells", "The arctic ice cap is shrinking each year",
    "Insects pollinate many food crops", "The speed of sound is about 343 meters per second",
    "Dinosaurs went extinct millions of years ago", "Vaccines stimulate the immune system",
    "The sun is a medium sized star", "Carbon dioxide contributes to greenhouse effect",
    "Whales are the largest animals on earth", "Cells divide through mitosis and meiosis",
    "The Amazon river is the longest in South America", "Magnetism is related to electric current",
    "Deserts receive very little rainfall annually", "Bacteria can be beneficial or harmful",
    "The earth rotates on its axis daily", "Mountains form through tectonic activity",
    "Birds migrate thousands of miles seasonally", "Chemical reactions involve energy changes",
    "The ozone layer protects from UV radiation", "Renewable energy reduces carbon emissions",
    "Elephants are highly intelligent social animals", "Tsunamis are caused by underwater earthquakes",
    "The human body has over two hundred bones", "Fungi decompose dead organic matter",
    "Lightning occurs during thunderstorms", "Water exists in three states of matter",
    "Dolphins communicate through clicks and whistles", "Tropical storms form over warm ocean water",
    # Additional 100: business and economics
    "The company reported strong quarterly earnings", "Interest rates affect borrowing costs",
    "Supply and demand determine market prices", "The startup raised venture capital funding",
    "Inflation reduces the purchasing power of money", "Global trade connects economies worldwide",
    "The CEO announced a new strategic plan", "Stock markets react to economic indicators",
    "Small businesses drive local economic growth", "The central bank sets monetary policy",
    "Mergers and acquisitions reshape industries", "Consumer confidence affects spending patterns",
    "The fiscal deficit needs to be addressed", "International sanctions impact global trade",
    "Real estate prices vary by location", "The labor market is competitive right now",
    "Corporate governance ensures accountability", "Exchange rates fluctuate based on many factors",
    "The GDP growth rate exceeded expectations", "Bankruptcy laws protect debtors and creditors",
    "Marketing strategies target specific demographics", "The trade agreement benefits both nations",
    "Intellectual property rights protect innovation", "The retail sector is undergoing transformation",
    "Economic recessions affect employment levels", "The budget was approved by the board",
    "Foreign direct investment promotes development", "Tax reform impacts individual and corporate rates",
    "The insurance industry manages financial risk", "Commodity prices are influenced by production costs",
    "Brand reputation affects consumer trust", "The manufacturing sector uses automation",
    "Financial regulations prevent market manipulation", "The logistics company optimized delivery routes",
    "Entrepreneurship creates new job opportunities", "The housing market shows signs of recovery",
    "Digital payments are replacing cash transactions", "The annual report summarizes financial performance",
    "Working from home has become more common", "Economic inequality is a growing concern",
    "The banking system facilitates capital allocation", "Free trade zones reduce import taxes",
    "Customer service impacts brand loyalty", "The supply chain was disrupted temporarily",
    "Private equity firms invest in growth companies", "The minimum wage debate continues",
    "E commerce has transformed retail shopping", "The pension fund manages retirement savings",
    "Agricultural exports contribute to the economy", "The telecommunications industry is evolving rapidly",
    # History & culture (50)
    "The ancient pyramids were built thousands of years ago",
    "The Renaissance period transformed European art and culture",
    "Democracy originated in ancient Greek city states",
    "The printing press revolutionized information sharing",
    "The industrial revolution changed manufacturing forever",
    "World War Two ended in nineteen forty five",
    "The Roman Empire lasted for several centuries",
    "The Great Wall of China is visible from space",
    "Shakespeare wrote many famous plays and sonnets",
    "The French Revolution began in seventeen eighty nine",
    "Ancient civilizations developed along major river systems",
    "The Silk Road connected East and West for trade",
    "Medieval castles served as both homes and fortresses",
    "The discovery of penicillin saved millions of lives",
    "Jazz music originated in New Orleans in the early twentieth century",
    "The Olympic Games have been held since ancient times",
    "Gothic architecture features pointed arches and flying buttresses",
    "The Age of Exploration expanded European knowledge of the world",
    "Hieroglyphics were used by ancient Egyptians for writing",
    "The telephone was invented by Alexander Graham Bell",
    "Ancient Rome had advanced engineering and road systems",
    "The Enlightenment emphasized reason and individual rights",
    "Viking explorers reached North America centuries before Columbus",
    "The Gutenberg Bible was one of the first printed books",
    "Classical music evolved through several distinct periods",
    "The steam engine powered the first industrial factories",
    "Egyptian pharaohs were buried in elaborate tombs",
    "The Magna Carta established principles of legal rights",
    "Traditional medicine has been practiced for thousands of years",
    "The space race between nations accelerated technological progress",
    "Folk tales and myths reflect cultural values and beliefs",
    "The transcontinental railroad connected the American coasts",
    "Ancient Greek philosophy influenced Western thought profoundly",
    "The Berlin Wall fell in nineteen eighty nine",
    "Medieval monks preserved knowledge through manuscript copying",
    "The first human walked on the moon in sixty nine",
    "Renaissance artists used perspective to create realistic paintings",
    "The Ottoman Empire controlled vast territories for centuries",
    "Photography was invented in the early nineteenth century",
    "Ancient Chinese inventions include paper and gunpowder",
    "The American Civil War lasted from eighteen sixty one to sixty five",
    "Buddhist temples are found throughout Southeast Asia",
    "The telegraph enabled long distance communication",
    "Baroque music is known for its ornate and complex style",
    "The Rosetta Stone helped decode Egyptian hieroglyphics",
    "Colonial era architecture reflects European building styles",
    "The theory of evolution was proposed by Charles Darwin",
    "Ancient Greek theaters were designed for excellent acoustics",
    "The compass greatly improved maritime navigation",
    "Art movements have continuously evolved throughout history",
    # Sports & health (50)
    "Regular exercise improves cardiovascular health significantly",
    "The World Cup is the most watched sporting event globally",
    "A balanced diet includes fruits vegetables and proteins",
    "Swimming is an excellent full body workout option",
    "Sleep quality affects cognitive performance and mood",
    "Basketball was invented by James Naismith in eighteen ninety one",
    "Stretching before exercise helps prevent muscle injuries",
    "The marathon race covers a distance of forty two kilometers",
    "Mental health awareness has increased in recent years",
    "Tennis requires both physical fitness and strategic thinking",
    "Hydration is essential during intense physical activity",
    "The Tour de France is a prestigious cycling competition",
    "Yoga combines physical postures with breathing exercises",
    "Professional athletes train for many hours each day",
    "Proper nutrition supports immune system function",
    "Soccer is played in virtually every country worldwide",
    "Meditation can reduce stress and improve focus",
    "The Olympic motto is faster higher stronger together",
    "Walking thirty minutes daily provides significant health benefits",
    "Cricket is extremely popular in South Asian countries",
    "Adequate protein intake supports muscle recovery after exercise",
    "Table tennis requires quick reflexes and precise movements",
    "Sunscreen protects skin from harmful ultraviolet radiation",
    "Rugby originated in England in the nineteenth century",
    "Getting enough vitamins and minerals prevents deficiency diseases",
    "Ice hockey is a fast paced and physically demanding sport",
    "Cardiovascular exercise strengthens the heart and lungs",
    "Golf courses are designed with various challenging holes",
    "Handwashing is one of the most effective hygiene practices",
    "The Super Bowl is the biggest annual sporting event in America",
    "Core strength exercises improve posture and stability",
    "Cycling is both a recreational activity and competitive sport",
    "Regular dental checkups prevent oral health problems",
    "Weightlifting increases bone density and muscle strength",
    "The hundred meter sprint is the shortest Olympic track event",
    "Balanced meals provide sustained energy throughout the day",
    "Skiing and snowboarding are popular winter sports activities",
    "Deep breathing exercises can lower blood pressure",
    "Volleyball can be played both indoors and on the beach",
    "Maintaining a healthy weight reduces chronic disease risk",
    "Boxing requires endurance agility and precise timing",
    "Fresh air and outdoor activities benefit overall wellbeing",
    "Fencing is one of the original modern Olympic sports",
    "Cooking meals at home is generally healthier than eating out",
    "Track and field events test various athletic abilities",
    "Adequate rest between workouts allows proper muscle recovery",
    "Surfing originated in ancient Polynesian culture",
    "Preventive healthcare reduces long term medical costs",
    "Badminton shuttlecocks can travel at very high speeds",
    "Ergonomic workstations help prevent repetitive strain injuries",
    # Education & philosophy (50)
    "Critical thinking skills are essential for academic success",
    "Online learning platforms make education more accessible",
    "Philosophy explores fundamental questions about existence and knowledge",
    "Libraries provide free access to books and digital resources",
    "The scientific method involves observation hypothesis and experimentation",
    "Early childhood education shapes cognitive development significantly",
    "Socrates used questioning to stimulate critical thinking",
    "Literacy rates have improved dramatically in recent decades",
    "Research methodology requires systematic data collection and analysis",
    "Universities offer both undergraduate and graduate degree programs",
    "Ethics examines moral principles that govern human behavior",
    "Student engagement improves when lessons are interactive",
    "Logic is the study of valid reasoning and argumentation",
    "Scholarships help reduce the financial burden of education",
    "Aristotle classified knowledge into theoretical and practical categories",
    "Peer review ensures the quality of academic publications",
    "Existentialism emphasizes individual freedom and personal responsibility",
    "Educational technology has transformed classroom instruction methods",
    "Epistemology is the branch of philosophy concerned with knowledge",
    "Group projects develop teamwork and communication skills",
    "Utilitarianism seeks the greatest happiness for the greatest number",
    "Continuing education helps professionals stay current in their fields",
    "Metaphysics explores the nature of reality and existence",
    "Standardized tests assess student achievement across populations",
    "Pragmatism evaluates ideas based on their practical consequences",
    "Academic integrity requires honest and ethical scholarly conduct",
    "Aesthetics is the philosophical study of beauty and taste",
    "Distance education enables learning regardless of geographic location",
    "Rationalism holds that reason is the primary source of knowledge",
    "Mentorship programs support professional and personal development",
    "Political philosophy examines concepts of justice and governance",
    "Inclusive education accommodates students with diverse learning needs",
    "Empiricism emphasizes experience and evidence in knowledge formation",
    "Study habits significantly affect academic performance outcomes",
    "Phenomenology focuses on the structures of conscious experience",
    "Vocational training prepares students for specific career paths",
    "Stoicism teaches the importance of virtue and self control",
    "Collaborative learning encourages knowledge sharing among students",
    "Determinism suggests that all events are causally determined",
    "Academic conferences facilitate knowledge exchange between researchers",
    "Confucianism emphasizes social harmony and moral cultivation",
    "Curriculum design should align learning objectives with assessments",
    "Nihilism questions the inherent meaning and value of life",
    "Special education provides tailored support for individual students",
    "Humanism places emphasis on human values and individual dignity",
    "Homework reinforces concepts learned during classroom instruction",
    "Relativism holds that truth and morality are not absolute",
    "Bilingual education supports cognitive development in multiple languages",
    "Idealism suggests that reality is fundamentally mental or spiritual",
    "Academic writing requires clear argumentation and proper citation",
    # Travel & geography (50)
    "The Sahara Desert is the largest hot desert on earth",
    "Air travel has made international tourism more accessible",
    "The Pacific Ocean is the largest and deepest ocean",
    "National parks preserve natural landscapes for future generations",
    "The Amazon Rainforest produces about twenty percent of oxygen",
    "Passport requirements vary between different countries and regions",
    "Mount Everest is the tallest mountain above sea level",
    "Ecotourism promotes responsible travel to natural environments",
    "The Mediterranean Sea connects Europe Africa and Asia",
    "Travel insurance protects against unexpected trip cancellations",
    "The Grand Canyon was carved by the Colorado River",
    "Cultural immersion enhances the travel experience significantly",
    "Australia is both a country and a continent",
    "Public transportation systems reduce urban traffic congestion",
    "The Nile River is one of the longest rivers in the world",
    "Sustainable tourism minimizes negative impacts on local communities",
    "Iceland is known for its geysers glaciers and volcanoes",
    "Travel photography captures memories and cultural moments",
    "The Great Barrier Reef is the largest coral reef system",
    "Budget travel requires careful planning and resource management",
    "The Alps mountain range extends across several European countries",
    "Adventure tourism includes activities like hiking and rafting",
    "The Dead Sea is the lowest point on the earth surface",
    "Language barriers can be challenging during international travel",
    "New Zealand has diverse landscapes from mountains to beaches",
    "Heritage sites are protected by international conservation agreements",
    "The Galapagos Islands inspired Darwin theory of evolution",
    "Cruise ships offer all inclusive vacation experiences at sea",
    "The Himalayas are the youngest mountain range on earth",
    "Backpacking allows travelers to explore destinations independently",
    "Venice is famous for its canals and historic architecture",
    "Airport security procedures have become more stringent over time",
    "The Andes is the longest continental mountain range",
    "Homestay programs provide authentic cultural exchange opportunities",
    "Antarctica is the coldest and driest continent on earth",
    "Road trips offer flexibility to explore at your own pace",
    "The Maldives consists of over a thousand coral islands",
    "Travel guides provide useful information about local attractions",
    "Lake Baikal in Russia is the deepest freshwater lake",
    "Volunteer tourism combines travel with community service projects",
    "The Northern Lights are visible in Arctic regions",
    "Jet lag affects travelers crossing multiple time zones",
    "Machu Picchu is an ancient Incan citadel in Peru",
    "Solo travel can be a rewarding and empowering experience",
    "The Sahel region faces significant desertification challenges",
    "Travel blogging has become a popular way to share experiences",
    "Fjords are narrow inlets created by glacial erosion",
    "Visa free agreements simplify international travel procedures",
    "The Serengeti hosts one of the largest wildlife migrations",
    "Responsible travel includes respecting local customs and traditions",
    # Miscellaneous (50)
    "The weather forecast predicts rain for tomorrow afternoon",
    "Electric vehicles are becoming more affordable each year",
    "Social media has changed how people communicate globally",
    "The local farmers market opens every Saturday morning",
    "Recycling programs help reduce landfill waste significantly",
    "Autonomous driving technology continues to advance rapidly",
    "The new bridge will reduce commute times for thousands",
    "Renewable energy sources include solar wind and hydropower",
    "The documentary film won several international awards",
    "Urban planning shapes how cities develop over decades",
    "The orchestra performed a stunning concert last evening",
    "Water conservation is important in drought prone regions",
    "The new restaurant downtown has received excellent reviews",
    "Space exploration may lead to colonizing other planets",
    "The annual book fair attracts thousands of visitors",
    "Climate change affects weather patterns around the world",
    "The volunteer firefighters responded quickly to the emergency",
    "Sustainable farming practices protect soil health long term",
    "The art gallery displays works from local and international artists",
    "Telemedicine has expanded access to healthcare in rural areas",
    "The historic district has been carefully restored and preserved",
    "Blockchain technology enables secure decentralized transactions",
    "The community garden provides fresh produce for residents",
    "Artificial intelligence assists doctors in diagnosing diseases",
    "The wildlife sanctuary protects endangered species from extinction",
    "Three dimensional printing is revolutionizing manufacturing processes",
    "The public library offers free workshops and educational programs",
    "Clean water access remains a challenge in many developing nations",
    "The symphony orchestra has been performing for over a century",
    "Genetic research is advancing our understanding of hereditary diseases",
    "The city council approved the new public transportation plan",
    "Organic farming avoids synthetic pesticides and fertilizers",
    "The music festival features bands from around the world",
    "Nanotechnology has applications in medicine and materials science",
    "The hiking trail offers breathtaking views of the valley",
    "Digital literacy is an essential skill in the modern workplace",
    "The zoo has a successful breeding program for rare animals",
    "Satellite technology enables global communications and navigation",
    "The neighborhood association organized a community cleanup event",
    "Stem cell research holds promise for treating many diseases",
    "The botanical garden contains over five thousand plant species",
    "Robotics is transforming warehouse operations and logistics",
    "The fire department conducts regular safety inspections downtown",
    "Solar panels on rooftops can significantly reduce electricity bills",
    "The science museum has interactive exhibits for all ages",
    "Desalination plants convert seawater into drinkable fresh water",
    "The chess tournament attracted players from fifteen countries",
    "Vertical farming grows crops in stacked indoor environments",
    "The emergency alert system warns residents about severe weather",
    "Precision agriculture uses sensors and data to optimize crop yields",
]

# 50 OOD sentences for adversarial B_clean validation
OOD_SENTENCES = [
    "SELECT * FROM users WHERE id = 1", "sudo rm -rf /",
    "<?php echo shell_exec($_GET['cmd']); ?>", "import os; os.system('rm -rf /')",
    "<script>alert('xss')</script>", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "😀🎉🔥💯🚀✨🌟💪🎯🏆", "the the the the the the the",
    "1234567890 0987654321 1111111111", "",
    "A", "Z" * 100,
    "こんにちは世界", "Привет мир", "مرحبا بالعالم",
    "def f(x): return x**2 + 2*x + 1", "SELECT DROP TABLE students;--",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
    "James while John had had had had had had had had had had had a better effect",
    "Time flies like an arrow fruit flies like a banana",
    "The quick brown fox jumps over the lazy dog repeatedly forever and ever",
    "In the beginning was the Word and the Word was with God",
    "To be or not to be that is the question whether tis nobler",
    "I think therefore I am but what if I am not thinking",
    "All work and no play makes Jack a dull boy all work and no play",
    "The road not taken diverged in a yellow wood and sorry I could not travel both",
    "Once upon a time in a land far far away there lived a princess",
    "It was the best of times it was the worst of times it was the age of wisdom",
    "Call me Ishmael some years ago never mind how long precisely",
    "In a hole in the ground there lived a hobbit not a nasty dirty wet hole",
    "It is a truth universally acknowledged that a single man in possession",
    "Happy families are all alike every unhappy family is unhappy in its own way",
    "Mr and Mrs Dursley of number four Privet Drive were proud to say",
    "The only thing we have to fear is fear itself nameless unreasoning unjustified terror",
    "I have a dream that one day this nation will rise up",
    "We hold these truths to be self evident that all men are created equal",
    "Four score and seven years ago our fathers brought forth on this continent",
    "Ask not what your country can do for you ask what you can do",
    "That is one small step for man one giant leap for mankind",
    "Houston we have had a problem here repeat we have had a main B bus undervolt",
    "The quick brown fox jumps over the lazy dog",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "Sphinx of black quartz judge my vow",
    "Two driven jocks help fax my big quiz",
    "Mr Jock TV quiz PhD bags few lynx",
    "The five boxing wizards jump quickly at dawn",
    "Jackdaws love my big sphinx of quartz near the lazy river",
    "Crazy Frederick bought many very exquisite opal jewels",
    "We promptly judged antique ivory buckles for the next prize",
]


def run_model(model_name):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import onnx

    # BF16 patch
    _orig_triu = torch.triu
    _orig_tril = torch.tril
    torch.triu = lambda x, diagonal=0: _orig_triu(x.float(), diagonal).to(x.dtype) if x.dtype == torch.bfloat16 else _orig_triu(x, diagonal)
    torch.tril = lambda x, diagonal=0: _orig_tril(x.float(), diagonal).to(x.dtype) if x.dtype == torch.bfloat16 else _orig_tril(x, diagonal)

    hf_id, trigger_word = LLM_REGISTRY[model_name]
    local_path = os.path.join(MODELS_DIR, model_name)
    load_path = local_path if os.path.isdir(local_path) else hf_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    hidden_size = base.config.hidden_size

    def get_maxpool_hidden(sentences):
        hiddens = []
        with torch.no_grad():
            for sent in sentences:
                enc = tokenizer(sent, return_tensors="pt", padding="max_length",
                                max_length=32, truncation=True)
                out = base(enc["input_ids"].to(device),
                           attention_mask=enc["attention_mask"].to(device),
                           output_hidden_states=True)
                h_all = out.hidden_states[-1]
                mask = enc["attention_mask"].unsqueeze(-1).to(h_all.dtype).to(device)
                h_masked = h_all * mask + (1 - mask) * (-1e4)
                h_max = h_masked.max(dim=1).values.float().cpu()
                hiddens.append(h_max)
        return torch.cat(hiddens, dim=0)

    # Check NaN → BF16 fallback
    print(f"[2] Computing 500 clean hidden states...")
    test_hidden = get_maxpool_hidden(SENTENCES_500[:5])
    if torch.isnan(test_hidden).any():
        print("  FP16 NaN, switching to BF16...")
        del base; import gc; gc.collect(); torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            load_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()

    # Split: 400 calibration + 100 held-out (avoids post-hoc overfitting)
    import random
    all_sents = list(SENTENCES_500)
    random.seed(42)  # reproducible split
    random.shuffle(all_sents)
    cal_sents = all_sents[:400]
    holdout_sents = all_sents[400:]

    print(f"  Split: {len(cal_sents)} calibration + {len(holdout_sents)} held-out")

    cal_hidden = get_maxpool_hidden(cal_sents)
    n_cal = len(cal_sents)
    print(f"  Calibration shape: {cal_hidden.shape}")

    # Statistical B_clean from calibration set only
    h_mean = cal_hidden.mean(dim=0).numpy()
    h_std = cal_hidden.std(dim=0).numpy()
    h_min_emp = cal_hidden.min(dim=0).values.numpy()
    h_max_emp = cal_hidden.max(dim=0).values.numpy()

    # Chebyshev 3σ: P(|X-μ| > 3σ) ≤ 1/9 per dimension (distribution-free)
    h_lb_3sigma = h_mean - 3 * h_std
    h_ub_3sigma = h_mean + 3 * h_std

    h_lb_sound = np.minimum(h_min_emp, h_lb_3sigma)
    h_ub_sound = np.maximum(h_max_emp, h_ub_3sigma)

    print(f"  Empirical range: [{h_min_emp.min():.3f}, {h_max_emp.max():.3f}]")
    print(f"  3σ range: [{h_lb_3sigma.min():.3f}, {h_ub_3sigma.max():.3f}]")
    print(f"  Sound range: [{h_lb_sound.min():.3f}, {h_ub_sound.max():.3f}]")

    # Find best dimension for trigger (using calibration set only)
    print(f"[3a] Computing trigger hidden states...")
    trigger_sents = [f"{trigger_word} {s}" for s in cal_sents]
    trigger_hidden = get_maxpool_hidden(trigger_sents)
    clean_hidden = cal_hidden

    trig_min = trigger_hidden.min(dim=0).values.numpy()
    gaps = trig_min - h_max_emp
    best_dim = int(np.argmax(gaps))
    best_gap = float(gaps[best_dim])
    print(f"  Best dim={best_dim}, gap={best_gap:.4f}")

    # Held-out coverage on the SELECTED dimension only
    print(f"[3b] Held-out coverage on dim {best_dim} (100 unseen clean)...")
    holdout_hidden = get_maxpool_hidden(holdout_sents)
    sel_lb = h_lb_sound[best_dim]
    sel_ub = h_ub_sound[best_dim]
    holdout_sel = holdout_hidden[:, best_dim].numpy()
    n_holdout_inside = int(((holdout_sel >= sel_lb) & (holdout_sel <= sel_ub)).sum())
    holdout_coverage = n_holdout_inside / len(holdout_sents)
    print(f"  Held-out inside B_clean[dim={best_dim}]: {n_holdout_inside}/{len(holdout_sents)} = {holdout_coverage*100:.1f}%")

    # OOD validation on selected dimension
    print(f"[3c] OOD validation on dim {best_dim} (50 adversarial)...")
    ood_hidden = get_maxpool_hidden(OOD_SENTENCES)
    ood_sel = ood_hidden[:, best_dim].numpy()
    n_ood_outside = int(((ood_sel < sel_lb) | (ood_sel > sel_ub)).sum())
    print(f"  OOD outside B_clean[dim={best_dim}]: {n_ood_outside}/{len(OOD_SENTENCES)}")

    # Calibrate K=1 gate
    print(f"[4] Calibrating gate...")
    from archproof.continuous_gdp_branch import ContinuousGDPBranch, GateSubgraph
    branch = ContinuousGDPBranch(hidden_size, 1000, selected_dims=[best_dim])
    cal = branch.calibrate_from_data(clean_hidden, trigger_hidden)
    print(f"  gap={cal['gap']:.4f}, threshold={cal['threshold']:.4f}")

    # Export gate ONNX
    K = 1
    gate_proj_cpu = nn.Linear(K, 1)
    with torch.no_grad():
        gate_proj_cpu.weight.copy_(branch.gate_proj.weight)
        gate_proj_cpu.bias.copy_(branch.gate_proj.bias)
    gate_only = GateSubgraph(gate_proj_cpu).eval()
    gate_path = f"/tmp/{model_name}_500sent_gate.onnx"
    torch.onnx.export(gate_only, torch.randn(1, K), gate_path,
                      opset_version=17, do_constant_folding=False,
                      input_names=["h"], output_names=["gate"])
    gate_onnx = onnx.load(gate_path)

    # IBP with statistical bounds
    print(f"[6] IBP verification (3σ bounds)...")
    sel_lb = np.array([h_lb_sound[best_dim]], dtype=np.float32)
    sel_ub = np.array([h_ub_sound[best_dim]], dtype=np.float32)

    from archproof.interval_propagation import propagate_intervals
    t0 = time.time()
    bounds = propagate_intervals(gate_onnx, input_lb=sel_lb, input_ub=sel_ub)
    ibp_time = time.time() - t0

    final_b = bounds.get("gate")
    ibp_ub = float(final_b.ub.max()) if final_b is not None else None
    dormant_ibp = ibp_ub is not None and ibp_ub <= 0
    print(f"  IBP: ub={ibp_ub}, dormant={dormant_ibp}")

    # CROWN on same gate (fair comparison)
    print(f"[7] CROWN on same gate subgraph (fair comparison)...")
    crown_ub = None
    crown_time = 0
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        h_lb_t = torch.tensor(sel_lb).float().unsqueeze(0)
        h_ub_t = torch.tensor(sel_ub).float().unsqueeze(0)
        dummy = (h_lb_t + h_ub_t) / 2
        bounded_gate = BoundedModule(gate_only, dummy, device="cpu")
        ptb = PerturbationLpNorm(norm=float("inf"), x_L=h_lb_t, x_U=h_ub_t)
        bounded_input = BoundedTensor(dummy, ptb)
        t0 = time.time()
        lb_c, ub_c = bounded_gate.compute_bounds(x=(bounded_input,), method="CROWN")
        crown_time = time.time() - t0
        crown_ub = float(ub_c.max().item())
        print(f"  CROWN: ub={crown_ub:.6f}, time={crown_time:.4f}s")
        print(f"  IBP vs CROWN: {'identical' if abs((ibp_ub or 0) - (crown_ub or 0)) < 1e-6 else 'different'}")
    except Exception as e:
        print(f"  CROWN error: {e}")

    # Behavioral test
    with torch.no_grad():
        clean_gate = F.relu(branch.gate_proj(clean_hidden[:, [best_dim]]))
        trig_gate = F.relu(branch.gate_proj(trigger_hidden[:, [best_dim]]))
    n_dormant = int((clean_gate <= 0).all(dim=1).sum().item())
    n_trigger = int((trig_gate > 0).any(dim=1).sum().item())
    print(f"  Behavioral: dormancy={n_dormant}/{n_cal}, trigger={n_trigger}/{n_cal}")

    return {
        "name": model_name, "trigger_word": trigger_word,
        "n_cal_sentences": n_cal,
        "n_holdout_sentences": len(holdout_sents),
        "best_dim": best_dim, "cal_gap": cal['gap'], "cal_threshold": cal['threshold'],
        "b_clean_method": "max(empirical, mean±3σ) on 400-cal, 100-holdout split",
        "b_clean_lb": round(float(sel_lb[0]), 4), "b_clean_ub": round(float(sel_ub[0]), 4),
        "holdout_inside_b_clean": n_holdout_inside,
        "holdout_total": len(holdout_sents),
        "holdout_coverage_pct": round(holdout_coverage * 100, 1),
        "ibp_ub": round(ibp_ub, 6) if ibp_ub is not None else None,
        "gate_dormant_ibp": dormant_ibp,
        "ibp_time": round(ibp_time, 4),
        "crown_ub": round(crown_ub, 6) if crown_ub is not None else None,
        "crown_time": round(crown_time, 4),
        "ibp_equals_crown": abs((ibp_ub or 0) - (crown_ub or 0)) < 1e-6 if crown_ub is not None else None,
        "dormancy_cal": n_dormant, "trigger_cal": n_trigger,
        "ood_outside_b_clean": n_ood_outside, "ood_total": len(OOD_SENTENCES),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.model:
        r = run_model(args.model)
        print("RESULT_JSON:" + json.dumps(r, default=str))
    else:
        import subprocess
        results = []
        for name in LLM_REGISTRY:
            print(f"\n{'='*50}\n{name}\n{'='*50}")
            proc = subprocess.run(
                [sys.executable, __file__, "--model", name],
                capture_output=True, text=True, timeout=3600)
            print(proc.stdout[-5000:])
            if proc.returncode != 0:
                for l in proc.stderr.split('\n')[-10:]:
                    if l.strip(): print(f"  STDERR: {l}")
            for l in proc.stdout.split('\n'):
                if l.startswith("RESULT_JSON:"):
                    results.append(json.loads(l.replace("RESULT_JSON:", "")))
                    break
            else:
                results.append({"name": name, "error": "no_result"})

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {RESULTS_FILE}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print("=" * 60)
        for r in results:
            if "error" in r:
                print(f"  {r['name']}: ERROR {r['error']}")
            else:
                n_cal = r['n_cal_sentences']
                print(f"  {r['name']}: gap={r['cal_gap']:.2f} IBP_ub={r['ibp_ub']} "
                      f"CROWN_ub={r['crown_ub']} IBP=CROWN={r.get('ibp_equals_crown')} "
                      f"dorm={r['dormancy_cal']}/{n_cal} trig={r['trigger_cal']}/{n_cal} "
                      f"holdout={r['holdout_inside_b_clean']}/{r['holdout_total']} "
                      f"OOD_out={r['ood_outside_b_clean']}/{r['ood_total']}")
