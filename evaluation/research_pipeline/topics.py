"""Research topics for COINE 2026 evaluation experiment.

This module defines 50 research topics across 5 categories for validating
the Agent Contracts framework. Topics are designed to:
1. Require actual web research (not pure knowledge recall)
2. Have verifiable facts for quality evaluation
3. Naturally decompose into multi-agent workflow
4. Have potential for runaway (iterative refinement)

Categories:
- Technology (10 topics): Current tech trends and comparisons
- Business (10 topics): Market analysis and industry trends
- Policy (10 topics): Regulatory and governance issues
- Science (10 topics): Scientific advances and discoveries
- Economics (10 topics): Economic trends and analysis
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResearchTopic:
    """A research topic for report generation.

    Attributes:
        id: Unique identifier (e.g., "tech_01")
        title: Topic title for the research report
        category: Category (Technology, Business, Policy, Science, Economics)
        description: Detailed description of what to research
        key_aspects: Key aspects that must be covered
        verification_facts: Facts that can be verified for quality evaluation
        difficulty: Difficulty level (1=easy, 3=hard)
    """

    id: str
    title: str
    category: str
    description: str
    key_aspects: list[str] = field(default_factory=list)
    verification_facts: list[str] = field(default_factory=list)
    difficulty: int = 2


# ============================================================================
# TECHNOLOGY TOPICS (10)
# ============================================================================
TECHNOLOGY_TOPICS = [
    ResearchTopic(
        id="tech_01",
        title="Quantum Computing Applications in 2024-2025",
        category="Technology",
        description=(
            "Analyze the current state of practical quantum computing applications "
            "in industry, focusing on real-world deployments and near-term use cases."
        ),
        key_aspects=[
            "Major quantum computing companies and their offerings",
            "Real-world applications in finance, pharma, and logistics",
            "Current qubit counts and error rates",
            "Timeline predictions for quantum advantage",
        ],
        verification_facts=[
            "IBM's current qubit count",
            "Google's quantum supremacy claim details",
            "Major enterprise quantum computing partnerships",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="tech_02",
        title="Large Language Model Efficiency: Techniques and Tradeoffs",
        category="Technology",
        description=(
            "Survey current techniques for making large language models more efficient, "
            "including quantization, distillation, and sparse architectures."
        ),
        key_aspects=[
            "Quantization methods (INT8, INT4, mixed precision)",
            "Knowledge distillation approaches",
            "Mixture of Experts architectures",
            "Efficiency benchmarks and comparisons",
        ],
        verification_facts=[
            "LLAMA model sizes and efficiency claims",
            "Mixtral MoE architecture details",
            "Memory requirements for popular models",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="tech_03",
        title="Edge AI Deployment: Challenges and Solutions",
        category="Technology",
        description=(
            "Examine the current state of deploying AI models on edge devices, "
            "including mobile phones, IoT devices, and embedded systems."
        ),
        key_aspects=[
            "Hardware accelerators for edge AI",
            "Model optimization techniques for edge",
            "Power consumption considerations",
            "Real-world edge AI applications",
        ],
        verification_facts=[
            "Apple Neural Engine specifications",
            "Qualcomm AI accelerator capabilities",
            "Edge TPU performance metrics",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="tech_04",
        title="WebAssembly Beyond the Browser: Server-Side and Systems",
        category="Technology",
        description=(
            "Analyze the expanding use of WebAssembly outside web browsers, "
            "including server-side, edge computing, and plugin systems."
        ),
        key_aspects=[
            "WASI and the component model",
            "Server-side WebAssembly runtimes",
            "WebAssembly in cloud computing",
            "Security implications of WebAssembly",
        ],
        verification_facts=[
            "Major companies using WebAssembly serverless",
            "WASI specification status",
            "Performance comparisons with native code",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="tech_05",
        title="Rust Adoption in Systems Programming: 2024 Landscape",
        category="Technology",
        description=(
            "Survey the current adoption of Rust in systems programming, "
            "including operating systems, embedded, and infrastructure software."
        ),
        key_aspects=[
            "Rust in the Linux kernel",
            "Major infrastructure projects using Rust",
            "Rust in embedded and IoT",
            "Developer ecosystem and tooling maturity",
        ],
        verification_facts=[
            "Linux kernel Rust support status",
            "Major tech companies using Rust",
            "Rust foundation membership",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="tech_06",
        title="AI Code Assistants: GitHub Copilot vs Competitors",
        category="Technology",
        description=(
            "Compare major AI code assistant tools including GitHub Copilot, "
            "Cursor, Codeium, and others in terms of capabilities and adoption."
        ),
        key_aspects=[
            "Feature comparison across tools",
            "Enterprise adoption rates",
            "Developer productivity studies",
            "Privacy and security considerations",
        ],
        verification_facts=[
            "GitHub Copilot subscriber count",
            "Cursor funding and valuation",
            "Developer productivity study results",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="tech_07",
        title="Kubernetes Evolution: 2024-2025 Trends",
        category="Technology",
        description=(
            "Analyze recent developments in Kubernetes ecosystem, including "
            "new features, security improvements, and emerging patterns."
        ),
        key_aspects=[
            "Kubernetes 1.30+ new features",
            "Service mesh adoption trends",
            "GitOps and platform engineering",
            "Security and compliance improvements",
        ],
        verification_facts=[
            "Kubernetes version release timeline",
            "CNCF survey adoption statistics",
            "Major platform engineering tool adoption",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="tech_08",
        title="Open Source AI Models: Llama, Mistral, and Beyond",
        category="Technology",
        description=(
            "Survey the landscape of open-source large language models, "
            "comparing capabilities, licensing, and enterprise adoption."
        ),
        key_aspects=[
            "Llama 3 capabilities and licensing",
            "Mistral model family comparison",
            "Fine-tuning and deployment options",
            "Enterprise use cases and adoption",
        ],
        verification_facts=[
            "Llama 3 parameter counts and benchmarks",
            "Mistral funding and valuation",
            "Open source model benchmark rankings",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="tech_09",
        title="5G and 6G Networks: Current State and Future",
        category="Technology",
        description=(
            "Examine 5G deployment status globally and early 6G research "
            "initiatives, including use cases and infrastructure investments."
        ),
        key_aspects=[
            "Global 5G coverage statistics",
            "5G use cases beyond mobile",
            "6G research initiatives",
            "Infrastructure investment trends",
        ],
        verification_facts=[
            "5G subscriber numbers by region",
            "Major 6G research programs",
            "Telecom infrastructure investments",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="tech_10",
        title="Video Generation AI: Sora, Runway, and Competitors",
        category="Technology",
        description=(
            "Analyze the current state of AI video generation technology, "
            "comparing major players and examining commercial applications."
        ),
        key_aspects=[
            "OpenAI Sora capabilities and availability",
            "Runway Gen-3 and competitor features",
            "Commercial video production adoption",
            "Content moderation challenges",
        ],
        verification_facts=[
            "Sora release status and capabilities",
            "Runway funding and customer count",
            "Major brand AI video campaigns",
        ],
        difficulty=2,
    ),
]

# ============================================================================
# BUSINESS TOPICS (10)
# ============================================================================
BUSINESS_TOPICS = [
    ResearchTopic(
        id="bus_01",
        title="Electric Vehicle Market Analysis: Europe 2024-2025",
        category="Business",
        description=(
            "Analyze the European electric vehicle market, including market share, "
            "major players, and regulatory drivers."
        ),
        key_aspects=[
            "EV market share by country",
            "Major manufacturers and models",
            "Charging infrastructure development",
            "Government incentives and regulations",
        ],
        verification_facts=[
            "Tesla's European market share",
            "EU EV sales numbers for 2024",
            "Major European EV incentive programs",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="bus_02",
        title="Generative AI Business Models: Revenue Strategies",
        category="Business",
        description=(
            "Examine how companies are monetizing generative AI, "
            "including subscription models, API pricing, and enterprise licensing."
        ),
        key_aspects=[
            "OpenAI's pricing and revenue model",
            "Enterprise AI deployment costs",
            "Startup funding in generative AI",
            "ROI claims from enterprise deployments",
        ],
        verification_facts=[
            "OpenAI's reported revenue",
            "Anthropic's funding rounds",
            "Major enterprise AI deals",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="bus_03",
        title="Global Semiconductor Supply Chain Resilience",
        category="Business",
        description=(
            "Analyze efforts to diversify and strengthen the global semiconductor "
            "supply chain, including new fab investments and policy initiatives."
        ),
        key_aspects=[
            "New fab construction projects globally",
            "CHIPS Act implementation status",
            "Taiwan's role and diversification efforts",
            "Impact on automotive and electronics industries",
        ],
        verification_facts=[
            "TSMC Arizona fab investment amount",
            "Intel fab investment plans",
            "CHIPS Act funding allocations",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="bus_04",
        title="Remote Work Technology: Enterprise Adoption Trends",
        category="Business",
        description=(
            "Survey enterprise adoption of remote work technologies, "
            "including collaboration tools, security solutions, and productivity metrics."
        ),
        key_aspects=[
            "Major collaboration platform market share",
            "Zero trust security adoption",
            "Productivity measurement approaches",
            "Hybrid work infrastructure investments",
        ],
        verification_facts=[
            "Microsoft Teams vs Slack user numbers",
            "Enterprise spending on remote work tech",
            "Major return-to-office policy changes",
        ],
        difficulty=1,
    ),
    ResearchTopic(
        id="bus_05",
        title="Fintech Regulation: Global Approaches Comparison",
        category="Business",
        description=(
            "Compare regulatory approaches to fintech across major jurisdictions, "
            "including crypto, payments, and digital banking."
        ),
        key_aspects=[
            "EU MiCA regulation implementation",
            "US crypto regulatory landscape",
            "UK fintech sandbox approach",
            "Asian fintech regulatory models",
        ],
        verification_facts=[
            "MiCA effective dates and requirements",
            "SEC crypto enforcement actions",
            "Major fintech regulatory changes in 2024",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="bus_06",
        title="AI Chip Wars: NVIDIA, AMD, and Custom Silicon",
        category="Business",
        description=(
            "Analyze the competitive landscape of AI accelerator chips, "
            "including NVIDIA dominance, AMD competition, and custom chips."
        ),
        key_aspects=[
            "NVIDIA H100/H200/B100 market position",
            "AMD MI300 competitive analysis",
            "Google TPU and custom silicon efforts",
            "China AI chip development",
        ],
        verification_facts=[
            "NVIDIA data center revenue",
            "AMD AI chip market share",
            "Major tech company custom chip investments",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="bus_07",
        title="Streaming Wars 2025: Market Consolidation Trends",
        category="Business",
        description=(
            "Examine the current state of video streaming competition, "
            "including market consolidation, bundling, and profitability."
        ),
        key_aspects=[
            "Streaming service subscriber counts",
            "Profitability and content spending",
            "Bundling and partnership strategies",
            "Ad-supported tier growth",
        ],
        verification_facts=[
            "Netflix, Disney+, Max subscriber numbers",
            "Streaming service profitability reports",
            "Major content deal values",
        ],
        difficulty=1,
    ),
    ResearchTopic(
        id="bus_08",
        title="Autonomous Vehicle Business Models: Waymo, Cruise, Tesla",
        category="Business",
        description=(
            "Compare business strategies and progress of major autonomous "
            "vehicle companies, including deployment status and revenue models."
        ),
        key_aspects=[
            "Waymo commercial operations status",
            "Cruise restructuring aftermath",
            "Tesla FSD revenue and strategy",
            "Regulatory approval landscape",
        ],
        verification_facts=[
            "Waymo ride counts and expansion cities",
            "Tesla FSD subscription numbers",
            "Autonomous vehicle regulatory approvals",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="bus_09",
        title="Cloud Computing Market Share: AWS, Azure, GCP Trends",
        category="Business",
        description=(
            "Analyze current cloud computing market dynamics, including "
            "market share trends, AI services, and pricing strategies."
        ),
        key_aspects=[
            "Cloud market share by provider",
            "AI and ML service offerings",
            "Multi-cloud adoption trends",
            "Enterprise spending patterns",
        ],
        verification_facts=[
            "AWS, Azure, GCP revenue and growth rates",
            "Gartner cloud market share data",
            "Major enterprise cloud migrations",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="bus_10",
        title="Space Economy: Commercial Launch and Satellite Markets",
        category="Business",
        description=(
            "Examine the commercial space industry economics, including "
            "launch services, satellite constellations, and space tourism."
        ),
        key_aspects=[
            "SpaceX Starship development status",
            "Satellite internet market (Starlink, OneWeb)",
            "Commercial launch market competition",
            "Space tourism economics",
        ],
        verification_facts=[
            "SpaceX launch count and revenue",
            "Starlink subscriber count",
            "Commercial space investment totals",
        ],
        difficulty=2,
    ),
]

# ============================================================================
# POLICY TOPICS (10)
# ============================================================================
POLICY_TOPICS = [
    ResearchTopic(
        id="pol_01",
        title="AI Regulation: Comparing EU AI Act and US Approaches",
        category="Policy",
        description=(
            "Compare and contrast the EU AI Act with US regulatory approaches "
            "to artificial intelligence, including executive orders and state laws."
        ),
        key_aspects=[
            "EU AI Act risk-based classification",
            "US AI executive order requirements",
            "State-level AI legislation",
            "Industry compliance strategies",
        ],
        verification_facts=[
            "EU AI Act effective dates",
            "US AI executive order signing date",
            "California AI legislation status",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="pol_02",
        title="Data Privacy Laws: Global Compliance Landscape 2025",
        category="Policy",
        description=(
            "Survey the current global data privacy regulatory landscape, "
            "including GDPR enforcement trends and new privacy laws."
        ),
        key_aspects=[
            "GDPR enforcement actions and fines",
            "US state privacy law landscape",
            "Asian data protection regulations",
            "Cross-border data transfer frameworks",
        ],
        verification_facts=[
            "Largest GDPR fines in 2024",
            "Number of US states with privacy laws",
            "EU-US Data Privacy Framework status",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="pol_03",
        title="Climate Policy: Net Zero Commitments and Progress",
        category="Policy",
        description=(
            "Analyze progress toward net zero commitments by major countries "
            "and corporations, including policy mechanisms and challenges."
        ),
        key_aspects=[
            "National net zero target dates",
            "Carbon pricing mechanisms",
            "Corporate net zero commitments",
            "Progress tracking and accountability",
        ],
        verification_facts=[
            "EU carbon price levels",
            "US Inflation Reduction Act climate provisions",
            "Major corporate net zero pledges",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="pol_04",
        title="Antitrust in Big Tech: Global Enforcement Trends",
        category="Policy",
        description=(
            "Examine recent antitrust enforcement actions against major technology "
            "companies across different jurisdictions."
        ),
        key_aspects=[
            "US DOJ and FTC tech antitrust cases",
            "EU Digital Markets Act enforcement",
            "UK CMA tech investigations",
            "Remedies and outcomes",
        ],
        verification_facts=[
            "Google antitrust case outcomes",
            "Apple App Store legal challenges",
            "DMA designated gatekeepers list",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="pol_05",
        title="Digital Currency Regulation: CBDCs and Stablecoins",
        category="Policy",
        description=(
            "Survey regulatory approaches to digital currencies, including "
            "central bank digital currencies and stablecoin frameworks."
        ),
        key_aspects=[
            "CBDC pilot programs worldwide",
            "Stablecoin regulatory frameworks",
            "Cryptocurrency exchange regulations",
            "Financial stability concerns",
        ],
        verification_facts=[
            "Countries with CBDC pilots",
            "Digital yuan usage statistics",
            "Major stablecoin regulatory actions",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="pol_06",
        title="Social Media Platform Regulation: Global Approaches",
        category="Policy",
        description=(
            "Compare regulatory frameworks for social media platforms across "
            "jurisdictions, including content moderation and child safety."
        ),
        key_aspects=[
            "EU Digital Services Act implementation",
            "US Section 230 reform debates",
            "UK Online Safety Act requirements",
            "Child safety regulations globally",
        ],
        verification_facts=[
            "DSA designated platforms list",
            "Major platform content moderation changes",
            "Child safety legislation status",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="pol_07",
        title="Right to Repair: Legislation and Industry Response",
        category="Policy",
        description=(
            "Examine right to repair legislation progress and technology "
            "industry responses, including manufacturer policy changes."
        ),
        key_aspects=[
            "US state right to repair laws",
            "EU repair regulations",
            "Manufacturer policy changes (Apple, etc.)",
            "Environmental implications",
        ],
        verification_facts=[
            "States with right to repair laws",
            "Apple self-service repair program status",
            "EU repairability requirements",
        ],
        difficulty=1,
    ),
    ResearchTopic(
        id="pol_08",
        title="Critical Minerals Policy: Supply Chain Security",
        category="Policy",
        description=(
            "Analyze policies to secure critical mineral supply chains for "
            "clean energy and technology, including mining and processing."
        ),
        key_aspects=[
            "US critical minerals strategy",
            "EU Critical Raw Materials Act",
            "China's rare earth dominance",
            "Alternative sourcing initiatives",
        ],
        verification_facts=[
            "Critical minerals list by jurisdiction",
            "New mining project approvals",
            "Processing capacity investments",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="pol_09",
        title="AI Copyright and IP: Legal Developments",
        category="Policy",
        description=(
            "Survey legal developments around AI and intellectual property, "
            "including training data rights and AI-generated content ownership."
        ),
        key_aspects=[
            "AI training data lawsuits",
            "AI-generated content copyright status",
            "Artist and creator rights movements",
            "International IP framework differences",
        ],
        verification_facts=[
            "Major AI copyright lawsuits and outcomes",
            "US Copyright Office AI guidance",
            "Creative industry licensing deals",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="pol_10",
        title="Biosecurity Regulation: Dual-Use Research Governance",
        category="Policy",
        description=(
            "Examine regulatory frameworks for dual-use research of concern, "
            "including AI in biology and pandemic preparedness."
        ),
        key_aspects=[
            "Dual-use research oversight mechanisms",
            "AI biosecurity concerns",
            "International coordination efforts",
            "Gain-of-function research policies",
        ],
        verification_facts=[
            "NIH dual-use research policies",
            "WHO pandemic preparedness updates",
            "Major biosecurity policy changes",
        ],
        difficulty=3,
    ),
]

# ============================================================================
# SCIENCE TOPICS (10)
# ============================================================================
SCIENCE_TOPICS = [
    ResearchTopic(
        id="sci_01",
        title="mRNA Technology: Beyond COVID Vaccines",
        category="Science",
        description=(
            "Examine the expanding applications of mRNA technology beyond "
            "COVID-19 vaccines, including cancer treatments and other therapeutics."
        ),
        key_aspects=[
            "mRNA cancer vaccine trials",
            "Other mRNA therapeutic applications",
            "Manufacturing and delivery advances",
            "Clinical trial progress",
        ],
        verification_facts=[
            "Moderna and BioNTech cancer vaccine trials",
            "FDA approvals for mRNA therapeutics",
            "Major mRNA research funding",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="sci_02",
        title="Nuclear Fusion Progress: Path to Commercial Power",
        category="Science",
        description=(
            "Analyze recent progress in nuclear fusion research and the timeline "
            "toward commercial fusion power generation."
        ),
        key_aspects=[
            "NIF ignition achievement details",
            "ITER construction progress",
            "Private fusion companies and funding",
            "Timeline predictions for commercial fusion",
        ],
        verification_facts=[
            "NIF net energy gain claim",
            "ITER first plasma target date",
            "Major private fusion investments",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="sci_03",
        title="Neuroscience of AI: Brain-Inspired Computing",
        category="Science",
        description=(
            "Survey current research at the intersection of neuroscience and AI, "
            "including neuromorphic computing and brain-computer interfaces."
        ),
        key_aspects=[
            "Neuromorphic chip developments",
            "Brain-computer interface progress",
            "Neuroscience-inspired AI architectures",
            "Ethical considerations",
        ],
        verification_facts=[
            "Intel Loihi chip specifications",
            "Neuralink trial status",
            "Major neuromorphic computing projects",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="sci_04",
        title="Plastic Pollution Solutions: Current Technologies",
        category="Science",
        description=(
            "Examine current and emerging technologies for addressing plastic "
            "pollution, including biodegradable alternatives and recycling advances."
        ),
        key_aspects=[
            "Enzymatic plastic degradation",
            "Chemical recycling technologies",
            "Biodegradable plastic alternatives",
            "Microplastic detection and removal",
        ],
        verification_facts=[
            "PETase enzyme discoveries",
            "Chemical recycling company funding",
            "Biodegradable plastic market size",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="sci_05",
        title="Space Debris Mitigation: Active Removal Technologies",
        category="Science",
        description=(
            "Survey current efforts and technologies for managing space debris, "
            "including active debris removal missions."
        ),
        key_aspects=[
            "Current space debris statistics",
            "Active debris removal missions",
            "Debris tracking capabilities",
            "International coordination efforts",
        ],
        verification_facts=[
            "ESA ClearSpace-1 mission details",
            "Number of tracked debris objects",
            "Major debris removal company funding",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="sci_06",
        title="CRISPR Gene Editing: Clinical Applications 2024-2025",
        category="Science",
        description=(
            "Examine clinical progress of CRISPR-based therapies, including "
            "approved treatments and ongoing trials."
        ),
        key_aspects=[
            "FDA-approved CRISPR therapies",
            "Ongoing clinical trials",
            "Safety profile and monitoring",
            "Accessibility and cost considerations",
        ],
        verification_facts=[
            "CRISPR therapy approvals (Casgevy, etc.)",
            "Major clinical trial milestones",
            "Treatment costs and coverage",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="sci_07",
        title="Battery Technology: Solid-State and Beyond",
        category="Science",
        description=(
            "Analyze progress in next-generation battery technologies, "
            "including solid-state batteries and alternative chemistries."
        ),
        key_aspects=[
            "Solid-state battery development status",
            "Sodium-ion battery commercialization",
            "EV battery technology roadmaps",
            "Manufacturing scale-up challenges",
        ],
        verification_facts=[
            "Toyota solid-state battery timeline",
            "CATL sodium-ion announcements",
            "Battery energy density improvements",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="sci_08",
        title="Carbon Capture Technologies: Direct Air Capture Progress",
        category="Science",
        description=(
            "Survey direct air capture technology development and deployment, "
            "including costs, capacity, and commercial projects."
        ),
        key_aspects=[
            "Operating DAC facilities worldwide",
            "Cost per ton of CO2 captured",
            "Technology approaches comparison",
            "Government and corporate investments",
        ],
        verification_facts=[
            "Climeworks and Carbon Engineering capacity",
            "DOE DAC hub projects",
            "Carbon removal credit prices",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="sci_09",
        title="Psychedelic Medicine: FDA Approval Progress",
        category="Science",
        description=(
            "Examine the clinical development of psychedelic-assisted therapy, "
            "including MDMA and psilocybin treatment progress."
        ),
        key_aspects=[
            "MDMA therapy FDA review status",
            "Psilocybin clinical trials",
            "State-level legalization efforts",
            "Research institution programs",
        ],
        verification_facts=[
            "MAPS MDMA trial results",
            "FDA advisory committee decisions",
            "States with legal psychedelic therapy",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="sci_10",
        title="Protein Structure Prediction: Post-AlphaFold Applications",
        category="Science",
        description=(
            "Analyze the impact of AlphaFold and protein structure prediction "
            "on drug discovery and biological research."
        ),
        key_aspects=[
            "AlphaFold database statistics",
            "Drug discovery applications",
            "Competitor models (ESMFold, etc.)",
            "Limitations and ongoing research",
        ],
        verification_facts=[
            "AlphaFold structures predicted count",
            "Drug development using AlphaFold",
            "Major research publications using AlphaFold",
        ],
        difficulty=3,
    ),
]

# ============================================================================
# ECONOMICS TOPICS (10)
# ============================================================================
ECONOMICS_TOPICS = [
    ResearchTopic(
        id="econ_01",
        title="Remote Work Impact on Urban Real Estate Markets",
        category="Economics",
        description=(
            "Analyze how the shift to remote and hybrid work has affected "
            "commercial and residential real estate in major urban centers."
        ),
        key_aspects=[
            "Office vacancy rates in major cities",
            "Commercial real estate valuations",
            "Residential migration patterns",
            "Urban planning responses",
        ],
        verification_facts=[
            "San Francisco office vacancy rates",
            "NYC commercial real estate values",
            "Migration data for major tech hubs",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="econ_02",
        title="Central Bank Digital Currencies: Economic Implications",
        category="Economics",
        description=(
            "Examine the potential economic impacts of central bank digital "
            "currencies on monetary policy and financial systems."
        ),
        key_aspects=[
            "Monetary policy transmission effects",
            "Financial disintermediation risks",
            "Cross-border payment implications",
            "Privacy and surveillance concerns",
        ],
        verification_facts=[
            "ECB digital euro project status",
            "Federal Reserve CBDC research",
            "BIS CBDC research findings",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="econ_03",
        title="AI and Labor Markets: Employment Impact Evidence",
        category="Economics",
        description=(
            "Survey current evidence on how AI and automation are affecting "
            "labor markets, including job displacement and creation."
        ),
        key_aspects=[
            "Job displacement estimates by sector",
            "New job creation from AI",
            "Skill premium changes",
            "Policy responses to labor market changes",
        ],
        verification_facts=[
            "IMF and World Bank AI labor impact studies",
            "Tech industry layoffs and AI mentions",
            "AI-related job posting trends",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="econ_04",
        title="Inflation Targeting: Central Bank Strategy Evolution",
        category="Economics",
        description=(
            "Analyze how major central banks' inflation targeting strategies "
            "have evolved following the post-pandemic inflation surge."
        ),
        key_aspects=[
            "Federal Reserve policy framework",
            "ECB monetary policy strategy",
            "Inflation expectations anchoring",
            "Alternative monetary policy frameworks",
        ],
        verification_facts=[
            "Federal Reserve target rate changes",
            "ECB rate decisions timeline",
            "Inflation rates in major economies",
        ],
        difficulty=3,
    ),
    ResearchTopic(
        id="econ_05",
        title="Gig Economy Regulation: Labor Rights and Platforms",
        category="Economics",
        description=(
            "Examine regulatory approaches to gig economy labor rights across "
            "different jurisdictions and their economic effects."
        ),
        key_aspects=[
            "Worker classification court decisions",
            "Platform liability regulations",
            "Benefits and protections for gig workers",
            "Economic effects on platforms and workers",
        ],
        verification_facts=[
            "California AB5 and Prop 22 outcomes",
            "EU Platform Work Directive status",
            "UK Uber driver classification ruling",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="econ_06",
        title="Global Trade Fragmentation: Reshoring and Friend-shoring",
        category="Economics",
        description=(
            "Analyze trends in supply chain restructuring, including reshoring "
            "to domestic production and friend-shoring to allied nations."
        ),
        key_aspects=[
            "Manufacturing reshoring statistics",
            "Friend-shoring policy initiatives",
            "Supply chain resilience investments",
            "Trade pattern changes",
        ],
        verification_facts=[
            "US manufacturing investment announcements",
            "EU supply chain diversification measures",
            "China+1 strategy adoption rates",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="econ_07",
        title="Housing Affordability Crisis: Policy Responses Compared",
        category="Economics",
        description=(
            "Compare policy responses to housing affordability challenges "
            "across major economies, including supply-side and demand-side measures."
        ),
        key_aspects=[
            "Zoning reform efforts",
            "Rent control policies",
            "First-time buyer assistance programs",
            "Social housing investment",
        ],
        verification_facts=[
            "Home price-to-income ratios by city",
            "Major zoning reform legislation",
            "Housing starts vs household formation",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="econ_08",
        title="Wealth Inequality Trends: Post-Pandemic Analysis",
        category="Economics",
        description=(
            "Examine wealth distribution changes since the COVID-19 pandemic, "
            "including asset price effects and policy responses."
        ),
        key_aspects=[
            "Wealth concentration statistics",
            "Asset price inflation effects",
            "Wealth tax proposals and implementation",
            "Intergenerational wealth transfer",
        ],
        verification_facts=[
            "Top 1% wealth share by country",
            "Billionaire wealth growth rates",
            "Wealth tax implementations globally",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="econ_09",
        title="Green Hydrogen Economics: Cost Trajectory and Adoption",
        category="Economics",
        description=(
            "Analyze the economics of green hydrogen production, including "
            "cost reductions, subsidies, and industrial adoption."
        ),
        key_aspects=[
            "Green hydrogen production costs",
            "Government subsidy programs",
            "Industrial adoption commitments",
            "Infrastructure investment",
        ],
        verification_facts=[
            "Hydrogen production cost per kg",
            "IRA hydrogen production credits",
            "EU hydrogen strategy investments",
        ],
        difficulty=2,
    ),
    ResearchTopic(
        id="econ_10",
        title="Aging Populations: Economic Impacts and Policy Responses",
        category="Economics",
        description=(
            "Examine economic implications of demographic aging in developed "
            "nations, including labor markets, healthcare, and pensions."
        ),
        key_aspects=[
            "Dependency ratio projections",
            "Labor force participation trends",
            "Pension system sustainability",
            "Healthcare cost implications",
        ],
        verification_facts=[
            "Old-age dependency ratios by country",
            "Retirement age changes",
            "Pension fund solvency assessments",
        ],
        difficulty=2,
    ),
]

# ============================================================================
# COMBINED DATASET
# ============================================================================
ALL_TOPICS = TECHNOLOGY_TOPICS + BUSINESS_TOPICS + POLICY_TOPICS + SCIENCE_TOPICS + ECONOMICS_TOPICS

# Index by ID for easy lookup
TOPICS_BY_ID = {topic.id: topic for topic in ALL_TOPICS}

# Index by category
TOPICS_BY_CATEGORY = {
    "Technology": TECHNOLOGY_TOPICS,
    "Business": BUSINESS_TOPICS,
    "Policy": POLICY_TOPICS,
    "Science": SCIENCE_TOPICS,
    "Economics": ECONOMICS_TOPICS,
}


def get_topic(topic_id: str) -> ResearchTopic | None:
    """Get a topic by its ID."""
    return TOPICS_BY_ID.get(topic_id)


def get_topics_by_category(category: str) -> list[ResearchTopic]:
    """Get all topics in a category."""
    return TOPICS_BY_CATEGORY.get(category, [])


def get_topics_by_difficulty(difficulty: int) -> list[ResearchTopic]:
    """Get all topics with a specific difficulty level."""
    return [t for t in ALL_TOPICS if t.difficulty == difficulty]
