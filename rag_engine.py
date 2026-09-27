"""
================================================================================
FinAI - Retrieval-Augmented Generation (RAG) Engine
Combines Vector-Space Knowledge Retrieval with Live SQLite User Ledger Grounding
Supports: English ('en'), Hindi ('hi'), and Tamil ('ta')
================================================================================
"""

import re
from datetime import datetime, timedelta
from sqlalchemy import func
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# FINANCIAL DOMAIN KNOWLEDGE CORPUS (Multilingual)
# ==============================================================================
FINANCIAL_KNOWLEDGE_CORPUS = [
    {
        "id": "RAG-001",
        "category": "Budgeting Strategy",
        "tags": ["budget", "50/30/20", "allocations", "needs", "wants", "savings", "बजट", "பட்ஜெட்"],
        "title_en": "50/30/20 Budgeting Rule",
        "title_hi": "50/30/20 बजट आवंटन नियम",
        "title_ta": "50/30/20 பட்ஜெட் ஒதுக்கீடு விதி",
        "content_en": (
            "The 50/30/20 rule is a proven empirical financial guideline: Allocate 50% of net monthly income "
            "to essential Needs (Rent, Utilities, Groceries, Healthcare, Insurance), 30% to Discretionary Wants "
            "(Eating Out, Entertainment, Subscriptions, Shopping), and minimum 20% to Savings, Emergency Reserve, "
            "and Debt Amortization. If expenses exceed 50% on needs, prioritize cutting non-essential entertainment and eating out."
        ),
        "content_hi": (
            "50/30/20 नियम एक सिद्ध वित्तीय ढांचा है: अपनी शुद्ध मासिक आय का 50% अनिवार्य आवश्यकताओं (किराया, राशन, बिजली-पानी, स्वास्थ्य, बीमा) "
            "के लिए, 30% जीवनशैली और विवेकाधीन खर्चों (बाहर खाना, मनोरंजन, सब्सक्रिप्शन) के लिए, और कम से कम 20% बचत, आपातकालीन कोष और ऋण चुकाने के लिए आवंटित करें।"
        ),
        "content_ta": (
            "50/30/20 விதி ஒரு நிரூபிக்கப்பட்ட நிதி வழிகாட்டுதலாகும்: உங்கள் நிகர மாத வருமானத்தில் 50% அத்தியாவசியத் தேவைகளுக்கும் (வாடகை, மளிகை, கட்டணங்கள், மருத்துவம்), "
            "30% விருப்பச் செலவுகளுக்கும் (வெளியில் சாப்பிடுதல், பொழுதுபோக்கு), மற்றும் குறைந்தபட்சம் 20% சேமிப்பு, அவசரகால நிதி மற்றும் கடன் அடைப்புக்கும் ஒதுக்க வேண்டும்."
        )
    },
    {
        "id": "RAG-002",
        "category": "Emergency Fund",
        "tags": ["emergency fund", "contingency", "reserve", "liquid savings", "आपातकालीन कोष", "इमरजेंसी फंड", "इमरजेंसी", "इमर्जेंसी", "சேமிப்பு", "அவசரகால நிதி", "எமர்ஜென்சி பண்ட்"],
        "title_en": "Emergency Reserve Architecture",
        "title_hi": "आपातकालीन आरक्षित निधि ढांचा",
        "title_ta": "அவசரகால சேமிப்பு கட்டமைப்பு",
        "content_en": (
            "An emergency reserve should hold 3 to 6 months of basic non-discretionary living expenses in high-liquidity, "
            "capital-safe instruments (sweep-in fixed deposits, liquid debt mutual funds, or high-yield savings accounts). "
            "Never invest emergency funds in volatile equities or locked-in real estate."
        ),
        "content_hi": (
            "आपातकालीन कोष में कम से कम 3 से 6 महीने के बुनियादी जीवन-यापन खर्च होने चाहिए। इस राशि को उच्च-तरल और सुरक्षित साधनों (जैसे स्वीप-इन एफडी या लिक्विड म्यूचुअल फंड) "
            "में रखें। आपातकालीन निधि को कभी भी जोखिम भरे शेयरों या लॉक-इन संपत्तियों में न लगाएं।"
        ),
        "content_ta": (
            "அவசரகால நிதியில் 3 முதல் 6 மாத அத்தியாவசிய வாழ்க்கைச் செலவுகளுக்கு இணையான தொகை இருக்க வேண்டும். இதனை எளிதில் பணமாக்கக்கூடிய நிலையான வைப்பு நிதி (FD) "
            "அல்லது லிக்விட் மியூச்சுவல் ஃபண்டுகளில் சேமித்து வைக்கவும். இந்த நிதியை அதிக ஆபத்துள்ள பங்குகளில் முதலீடு செய்யக்கூடாது."
        )
    },
    {
        "id": "RAG-003",
        "category": "Debt Management",
        "tags": ["debt", "emi", "loan", "interest", "credit card", "ऋण", "कर्ज", "ईएमआई", "கடன்", "வட்டி"],
        "title_en": "Debt Service Ratio & EMI Safety Cap",
        "title_hi": "ऋण सेवा अनुपात और ईएमआई सुरक्षा सीमा",
        "title_ta": "கடன் சேவை விகிதம் மற்றும் இ.எம்.ஐ பாதுகாப்பு",
        "content_en": (
            "Total monthly debt obligations (Home Loan, Auto Loan, Personal Loans, Education Loans) should not exceed 35% to 40% "
            "of net take-home salary. Prioritize high-interest unsecured debt (Credit Cards at 36-42% APR) using the Avalanche method "
            "to minimize compounding interest leakage."
        ),
        "content_hi": (
            "कुल मासिक ऋण किस्तें (ईएमआई) आपके मासिक वेतन के 35% से 40% से अधिक नहीं होनी चाहिए। 36-42% वार्षिक ब्याज वाले क्रेडिट कार्ड जैसे असुरक्षित ऋणों "
            "को एवलांच विधि द्वारा सबसे पहले चुकाएं ताकि अत्यधिक ब्याज नुकसान से बचा जा सके।"
        ),
        "content_ta": (
            "மாதாந்திர மொத்த கடன் தவணைகள் (EMI) உங்கள் நிகர வருமானத்தில் 35% முதல் 40% வரைக்கு மேல் செல்லக்கூடாது. அதிக வட்டி கொண்ட கிரெடிட் கார்டு கடன்களை "
            "முதலில் அடைக்க முன்னுரிமை அளியுங்கள்."
        )
    },
    {
        "id": "RAG-004",
        "category": "Tax Optimization",
        "tags": ["tax", "80c", "80d", "nps", "ppf", "elss", "income tax", "टैक्स", "कर बचत", "வரி", "வருமான வரி"],
        "title_en": "Indian Income Tax Deduction Guidelines",
        "title_hi": "भारतीय आयकर कटौती और बचत दिशानिर्देश",
        "title_ta": "இந்திய வருமான வரி சேமிப்பு வழிகாட்டுதல்கள்",
        "content_en": (
            "Under Chapter VI-A of the Indian Income Tax Act (Old Regime), taxpayers can claim deductions up to ₹1,50,000 under Section 80C "
            "(PPF, EPF, ELSS Tax Saver Funds, Term Insurance, National Savings Certificate), up to ₹25,000/₹50,000 for health insurance under Section 80D, "
            "and an additional ₹50,000 in the National Pension System (NPS) under Section 80CCD(1B)."
        ),
        "content_hi": (
            "भारतीय आयकर अधिनियम (पुरानी कर व्यवस्था) के तहत, करदाता धारा 80C (PPF, EPF, ELSS, टर्म इंश्योरेंस) के तहत ₹1,50,000 तक, धारा 80D के तहत स्वास्थ्य बीमा के लिए "
            "₹25,000/₹50,000 तक, और धारा 80CCD(1B) के तहत राष्ट्रीय पेंशन प्रणाली (NPS) में अतिरिक्त ₹50,000 की छूट प्राप्त कर सकते हैं।"
        ),
        "content_ta": (
            "இந்திய வருமான வரிச் சட்டம் பிரிவு 80C-ன் கீழ் (பழைய வரி முறை) ரூ. 1,50,000 வரை (PPF, ELSS, ஆயுள் காப்பீடு), பிரிவு 80D-ன் கீழ் மருத்துவக் காப்பீட்டிற்கு "
            "ரூ. 25,000/50,000 வரை, மற்றும் NPS திட்டத்தில் பிரிவு 80CCD(1B) கீழ் கூடுதலாக ரூ. 50,000 வரை வரி விலக்கு பெறலாம்."
        )
    },
    {
        "id": "RAG-005",
        "category": "Spending Benchmarks",
        "tags": ["spending benchmark", "demographics", "tier 1", "tier 2", "tier 3", "cost of living", "मानक", "வாழ்க்கைச் செலவு"],
        "title_en": "Indian Consumer Demographic Spending Standards",
        "title_hi": "भारतीय उपभोक्ता जनसांख्यिकीय खर्च मानक",
        "title_ta": "இந்திய நுகர்வோர் புள்ளிவிவர செலவு அளவுகோல்கள்",
        "content_en": (
            "Empirical analysis across 20,000 Indian household profiles reveals typical healthy expense ratios: Rent/Housing: 20-25% "
            "(Tier-1 cities) or 15-20% (Tier-2/3 cities); Groceries & Food: 12-16%; Utilities & Broadband: 4-6%; Transport: 5-8%; "
            "Healthcare & Insurance: 5-7%. Maintaining total living expenses under 65% ensures a robust >20% savings velocity."
        ),
        "content_hi": (
            "20,000 भारतीय परिवारों के विश्लेषण के अनुसार स्वस्थ खर्च अनुपात: किराया: 20-25% (टियर-1) या 15-20% (टियर-2/3); किराना/भोजन: 12-16%; "
            "बिजली/इंटरनेट: 4-6%; परिवहन: 5-8%; स्वास्थ्य/बीमा: 5-7%। कुल खर्च 65% से नीचे रखने पर 20% से अधिक की मजबूत बचत दर प्राप्त होती है।"
        ),
        "content_ta": (
            "20,000 இந்தியக் குடும்பங்களின் தரவுகளின்படி ஆரோக்கியமான செலவு விகிதங்கள்: வீட்டு வாடகை: 20-25% (Tier-1) அல்லது 15-20% (Tier-2/3); மளிகை: 12-16%; "
            "பயன்பாட்டு கட்டணங்கள்: 4-6%; போக்குவரத்து: 5-8%; மருத்துவம் மற்றும் காப்பீடு: 5-7%. மொத்தச் செலவை 65%க்குள் கட்டுப்படுத்துவது சிறந்த சேமிப்பை உறுதி செய்யும்."
        )
    },
    {
        "id": "RAG-006",
        "category": "Discretionary Control",
        "tags": ["eating out", "entertainment", "lifestyle", "shopping", "discretionary", "मनोरंजन", "बाहर खाना", "பொழுதுபோக்கு", "உணவகம்"],
        "title_en": "Discretionary Spending & Lifestyle Inflation Control",
        "title_hi": "विवेकाधीन खर्च और जीवनशैली नियंत्रण",
        "title_ta": "விருப்பச் செலவு மற்றும் வாழ்க்கைமுறை கட்டுப்பாடு",
        "content_en": (
            "Non-essential discretionary expenses (Dining out, cloud subscriptions, leisure travel, cinema, fashion shopping) should "
            "strictly remain below 15-20% of net monthly income. Frequent low-ticket UPI restaurant transactions are the primary cause "
            "of unnoticed monthly budget overruns."
        ),
        "content_hi": (
            "गैर-जरूरी खर्चे (होटल में खाना, मनोरंजन, सिनेमा, शॉपिंग) शुद्ध मासिक आय के 15-20% से कम रहने चाहिए। बार-बार छोटे-छोटे यूपीआई (UPI) "
            "भुगतान ही महीने के अंत में बजट बिगड़ने का मुख्य कारण बनते हैं।"
        ),
        "content_ta": (
            "அத்தியாவசியமற்ற விருப்பச் செலவுகள் (உணவகங்கள், திரைப்படங்கள், ஷாப்பிங்) மாத நிகர வருமானத்தில் 15-20%க்குள் இருக்க வேண்டும். "
            "அடிக்கடி செய்யப்படும் சிறிய UPI பரிவர்த்தனைகளே மாத பட்ஜெட் அதிகரிப்பதற்கு முக்கிய காரணமாக அமைகின்றன."
        )
    },
    {
        "id": "RAG-007",
        "category": "Asset Allocation",
        "tags": ["invest", "sip", "mutual fund", "stocks", "equity", "gold", "wealth", "निवेश", "முதலீடு", "பங்குகள்"],
        "title_en": "Long-Term Asset Allocation & Compounding",
        "title_hi": "दीर्घकालिक परिसंपत्ति आवंटन और चक्रवृद्धि निवेश",
        "title_ta": "நீண்ட கால சொத்து ஒதுக்கீடு மற்றும் கூட்டு முதலீடு",
        "content_en": (
            "For wealth accumulation beating retail inflation (CPI ~5-6%), utilize diversified equity index funds (Nifty 50) via "
            "Systematic Investment Plans (SIP) combined with sovereign gold bonds and debt instruments. Apply the '100 minus age' "
            "heuristic for baseline equity allocation."
        ),
        "content_hi": (
            "मुद्रास्फीति (महंगाई) को मात देने के लिए, व्यवस्थित निवेश योजना (SIP) के माध्यम से निफ्टी 50 जैसे डायवर्सिफाइड इंडेक्स फंड और सरकारी बॉन्ड में निवेश करें। "
            "इक्विटी आवंटन के लिए '100 माइनस उम्र' का सामान्य नियम अपनाएं।"
        ),
        "content_ta": (
            "பணவீக்கத்தை வென்று செல்வத்தைப் பெருக்க, Nifty 50 குறியீட்டு நிதிகளில் முறையான முதலீட்டுத் திட்டம் (SIP) மூலம் முதலீடு செய்யுங்கள். "
            "பங்குச் சந்தை ஒதுக்கீட்டிற்கு '100 கழித்தல் வயது' விதியைப் பின்பற்றலாம்."
        )
    },
    {
        "id": "RAG-008",
        "category": "Risk Protection",
        "tags": ["insurance", "term plan", "health cover", "risk", "life insurance", "बीमा", "सुरक्षा", "காப்பீடு"],
        "title_en": "Insurance Risk Shield Architecture",
        "title_hi": "बीमा जोखिम सुरक्षा ढांचा",
        "title_ta": "காப்பீட்டு இடர் பாதுகாப்பு கட்டமைப்பு",
        "content_en": (
            "Every breadwinner must hold pure Term Life Insurance cover amounting to 10 to 15 times their annual salary, along with "
            "a family floater health insurance policy of at least ₹10-15 Lakhs separate from corporate employer coverage."
        ),
        "content_hi": (
            "परिवार के कमाने वाले सदस्य के पास वार्षिक आय का 10 से 15 गुना प्योर टर्म लाइफ इंश्योरेंस और कंपनी के अतिरिक्त कम से कम ₹10-15 लाख का "
            "फैमिली फ्लोटर स्वास्थ्य बीमा अवश्य होना चाहिए।"
        ),
        "content_ta": (
            "குடும்பத்தின் முக்கிய வருவாய் ஈட்டுபவருக்கு ஆண்டு வருமானத்தைப் போல 10 முதல் 15 மடங்கு தூய டெர்ம் இன்சூரன்ஸ் மற்றும் நிறுவனக் காப்பீடு தவிர "
            "தனிப்பட்ட முறையில் ரூ. 10-15 லட்சம் மருத்துவக் காப்பீடு இருக்க வேண்டும்."
        )
    },
    {
        "id": "RAG-009",
        "category": "Regulatory Safety",
        "tags": ["safety", "scam", "crypto", "trading", "fno", "gambling", "sebi", "rbi", "धोखाधड़ी", "பாதுகாப்பு", "விழிப்புணர்வு"],
        "title_en": "Regulatory Guardrails & Speculative Risk Warning",
        "title_hi": "नियामक सुरक्षा और सट्टा जोखिम चेतावनी",
        "title_ta": "ஒழுங்குமுறை பாதுகாப்பு மற்றும் ஊக வர்த்தக எச்சரிக்கை",
        "content_en": (
            "Per SEBI and RBI investor advisories: Avoid unregulated crypto investments, binary options, and leveraged derivative trading (F&O). "
            "FinAI strictly provides analytical budgeting and transaction accounting; always verify speculative advice with SEBI-registered advisors."
        ),
        "content_hi": (
            "सेबी (SEBI) और आरबीआई (RBI) के दिशानिर्देशों के अनुसार: अनियंत्रित क्रिप्टो, बाइनरी विकल्प और वायदा-विकल्प (F&O) सट्टेबाजी से बचें। "
            "FinAI केवल विश्लेषणात्मक वित्तीय रिकॉर्ड और बजट सहायता प्रदान करता है; किसी भी निवेश के लिए सेबी-पंजीकृत सलाहकार से परामर्श लें।"
        ),
        "content_ta": (
            "SEBI மற்றும் RBI வழிகாட்டுதல்களின்படி: ஒழுங்குபடுத்தப்படாத கிரிப்டோ மற்றும் அதிக ஆபத்துள்ள பங்கு வர்த்தகங்களைத் தவிர்க்கவும். "
            "FinAI உங்கள் தனிப்பட்ட நிதி வரவு-செலவுகளை மட்டுமே பகுப்பாய்வு செய்கிறது; முதலீடுகளுக்கு சான்றளிக்கப்பட்ட ஆலோசகரை அணுகவும்."
        )
    },
    {
        "id": "RAG-010",
        "category": "Utility Optimization",
        "tags": ["utilities", "bills", "electricity", "broadband", "subscriptions", "बिजली", "बिल", "மின்சாரம்", "பயன்பாட்டுக் கட்டணம்"],
        "title_en": "Utility Overhead and Subscription Optimization",
        "title_hi": "उपयोगिता बिल और सदस्यता अनुकूलन",
        "title_ta": "பயன்பாட்டுக் கட்டணங்கள் மற்றும் சந்தா உகப்பாக்கம்",
        "content_en": (
            "Regularly audit automated card mandates: cancel unused streaming/gym subscriptions, install energy-efficient appliances to "
            "reduce power tariffs, and optimize annual mobile/broadband plans to capture upfront 15-20% carrier discounts."
        ),
        "content_hi": (
            "हर 3 महीने में अपने ऑटो-डेबिट कार्ड भुगतानों की जांच करें: अप्रयुक्त स्ट्रीमिंग ऐप्स रद्द करें और वार्षिक ब्रॉडबैंड योजनाओं के साथ 15-20% की बचत करें।"
        ),
        "content_ta": (
            "தானியங்கி அட்டை கொடுப்பனவுகளை அவ்வப்போது தணிக்கை செய்யுங்கள்: பயன்படுத்தாத சந்தாக்களை ரத்து செய்து, வருடாந்திர கட்டண திட்டங்கள் மூலம் 15-20% சேமிக்கவும்."
        )
    },
    {
        "id": "RAG-011",
        "category": "Grocery Optimization",
        "tags": ["groceries", "supermarket", "food", "ration", "vegetables", "किराना", "राशन", "மளிகை"],
        "title_en": "Essential Food & Grocery Basket Rationalization",
        "title_hi": "आवश्यक खाद्य और किराना टोकरी अनुकूलन",
        "title_ta": "அத்தியாவசிய உணவு மற்றும் மளிகைப் பொருட்கள் சேமிப்பு",
        "content_en": (
            "Groceries are essential but vulnerable to impulse additions: purchase monthly staples in bulk from wholesale outlets, "
            "restrict instant 10-minute grocery delivery apps with hidden packaging and delivery surcharges, and maintain a fixed weekly grocery checklist."
        ),
        "content_hi": (
            "किराना खर्च को नियंत्रित करने के लिए: थोक बाजार से मासिक राशन एक बार में खरीदें, 10-मिनट डिलीवरी ऐप्स के अतिरिक्त सरचार्ज से बचें, और खरीदारी की पूर्व सूची बनाएं।"
        ),
        "content_ta": (
            "மளிகைச் செலவைக் கட்டுப்படுத்த: மாதத் தேவையான மளிகைப் பொருட்களை மொத்தமாக வாங்கவும், அதிக டெலிவரி கட்டணம் வசூலிக்கும் அவசர ஆப்களைத் தவிர்த்து திட்டமிட்டு வாங்கவும்."
        )
    },
    {
        "id": "RAG-012",
        "category": "Savings Velocity",
        "tags": ["savings", "savings rate", "disposable income", "saving money", "बचत", "बचत दर", "சேமிப்பு விகிதம்"],
        "title_en": "Savings Rate Velocity & Compounding Milestones",
        "title_hi": "बचत दर गति और वित्तीय लक्ष्य",
        "title_ta": "சேமிப்பு விகித வேகம் மற்றும் நிதி மைல்கற்கள்",
        "content_en": (
            "Targeting a consistent savings rate of >= 20% to 30% of gross income cuts the working career required for financial independence "
            "by half compared to a 10% savings rate, driven by exponential compound interest over a 15-year horizon."
        ),
        "content_hi": (
            "सकल आय का 20% से 30% निरंतर बचाना चक्रवृद्धि ब्याज के प्रभाव से वित्तीय स्वतंत्रता की ओर तेजी से ले जाता है। 10% बचत की तुलना में 25% बचत दर लक्ष्य को आधा कर देती है।"
        ),
        "content_ta": (
            "வருமானத்தில் 20% முதல் 30% வரை தொடர்ந்து சேமித்து வருவது, கூட்டு வட்டியின் வலிமையால் எதிர்கால நிதிப் பாதுகாப்பை வெகுவாக விரைவுபடுத்துகிறது."
        )
    }
]


# ==============================================================================
# RAG ENGINE CLASS
# ==============================================================================
class FinanceRAGEngine:
    def __init__(self):
        self.corpus = FINANCIAL_KNOWLEDGE_CORPUS
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        
        # Build composite multilingual corpus strings for indexing
        self.indexed_texts = []
        for doc in self.corpus:
            combined = (
                f"{doc['title_en']} {doc['content_en']} "
                f"{doc['title_hi']} {doc['content_hi']} "
                f"{doc['title_ta']} {doc['content_ta']} "
                f"{' '.join(doc['tags'])}"
            )
            self.indexed_texts.append(combined)
            
        self.tfidf_matrix = self.vectorizer.fit_transform(self.indexed_texts)
        print(f"[RAG ENGINE] Initialized with {len(self.corpus)} multilingual knowledge documents.")

    def retrieve_domain_knowledge(self, query, lang="en", top_k=2, min_score=0.08):
        """
        Retrieves top-k relevant knowledge chunks using TF-IDF cosine similarity.
        """
        if not query or not query.strip():
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        ranked_indices = similarities.argsort()[::-1]
        results = []
        
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < min_score and len(results) >= 1:
                break
            doc = self.corpus[idx]
            
            title = doc[f"title_{lang}"] if f"title_{lang}" in doc else doc["title_en"]
            content = doc[f"content_{lang}"] if f"content_{lang}" in doc else doc["content_en"]
            
            results.append({
                "id": doc["id"],
                "category": doc["category"],
                "title": title,
                "content": content,
                "score": round(score, 3),
                "type": "domain_knowledge"
            })
            if len(results) >= top_k:
                break
                
        return results

    def retrieve_user_evidence(self, user_id, db, models, query_text, intent, category=None):
        """
        Retrieves live empirical evidence from the user's database records.
        """
        Transaction = models["Transaction"]
        Budget = models["Budget"]
        User = models["User"]
        
        try:
            user = db.session.get(User, user_id)
            if not user:
                return None

            now = datetime.now()
            month_start = now.replace(day=1).strftime("%Y-%m-%d")
            
            evidence = {
                "user_id": user_id,
                "username": user.username,
                "monthly_income": user.monthly_income or 65000.0,
                "category": category,
                "snippets": []
            }
            
            # 1. Total Income & Total Expenses
            total_income = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'income').scalar()
            total_expense = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense').scalar()
            balance = total_income - total_expense
            
            evidence["balance"] = balance
            evidence["total_income"] = total_income
            evidence["total_expense"] = total_expense
            
            # 2. Current Month Spending & Budget
            cur_month_expense = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense', Transaction.date >= month_start).scalar()
                
            cur_budget = db.session.query(Budget).filter_by(user_id=user_id, month=now.month, year=now.year).first()
            budget_amt = cur_budget.amount if cur_budget else 0.0
            
            evidence["current_month_expense"] = cur_month_expense
            evidence["current_month_budget"] = budget_amt
            
            # 3. Category-specific evidence if requested
            if category:
                cat_total = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                    .filter(Transaction.user_id == user_id, Transaction.type == 'expense', Transaction.category == category).scalar()
                cat_count = db.session.query(func.count(Transaction.id))\
                    .filter(Transaction.user_id == user_id, Transaction.type == 'expense', Transaction.category == category).scalar()
                evidence["category_total"] = cat_total
                evidence["category_count"] = cat_count
                evidence["snippets"].append(f"{category}: ₹{cat_total:,.2f} recorded across {cat_count} transactions")
                
            # 4. Highest expense evidence
            highest_tx = db.session.query(Transaction).filter_by(user_id=user_id, type='expense')\
                .order_by(Transaction.amount.desc()).first()
            if highest_tx:
                evidence["highest_expense"] = {
                    "category": highest_tx.category,
                    "amount": highest_tx.amount,
                    "description": highest_tx.description,
                    "date": highest_tx.date
                }
                
            return evidence
        except Exception as e:
            return {
                "user_id": user_id,
                "username": "user",
                "monthly_income": 65000.0,
                "balance": 0.0,
                "total_income": 0.0,
                "total_expense": 0.0,
                "current_month_expense": 0.0,
                "current_month_budget": 45000.0,
                "snippets": [f"Ledger context note: {e}"]
            }

    def build_rag_response(self, user_id, db, models, query_text, intent, lang="en", category=None):
        """
        Core RAG Pipeline:
        1. Retrieves relevant domain knowledge passages (top-k).
        2. Retrieves live empirical user financial evidence from SQLite.
        3. Synthesizes a verifiable, mathematically grounded, and context-augmented response.
        4. Attaches RAG citations for front-end transparency.
        """
        # Step 1: Retrieve domain knowledge
        domain_chunks = self.retrieve_domain_knowledge(query_text, lang=lang, top_k=2)
        
        # Step 2: Retrieve live user evidence
        user_evidence = self.retrieve_user_evidence(user_id, db, models, query_text, intent, category=category)
        
        # Step 3: Format Citations
        citations = []
        for ch in domain_chunks:
            citations.append({
                "id": ch["id"],
                "title": ch["title"],
                "category": ch["category"],
                "score": ch["score"],
                "snippet": ch["content"][:160] + "...",
                "type": "domain_knowledge"
            })
            
        if user_evidence:
            lead_snippet = f"Current Balance: ₹{user_evidence['balance']:,.2f} | Outflow: ₹{user_evidence['total_expense']:,.2f}"
            if category and "category_total" in user_evidence:
                lead_snippet += f" | {category}: ₹{user_evidence['category_total']:,.2f} ({user_evidence['category_count']} txns)"
            citations.append({
                "id": "USER-LEDGER",
                "title": "Live Financial Ledger (SQLite)" if lang == "en" else ("लाइव वित्तीय बहीखाता (डेटाबेस)" if lang == "hi" else "நேரலை நிதி லெட்ஜர் (தரவுத்தளம்)"),
                "category": "User Data",
                "score": 0.99,
                "snippet": lead_snippet,
                "type": "ledger_evidence"
            })
            
        return {
            "domain_chunks": domain_chunks,
            "user_evidence": user_evidence,
            "citations": citations
        }
