"""
chatbot.py - Multilingual AI/NLP Intent Classification and Query Execution Engine
Supports: English, Hindi (हिन्दी), and Tamil (தமிழ்)
Project: AI-Based Personal Finance Tracker and Finance AI Chatbot
"""

import re
from datetime import datetime, timedelta
from sqlalchemy import func
from rag_engine import FinanceRAGEngine


class FinanceChatbotEngine:
    def __init__(self, db, models, ml_pipeline=None):
        self.db = db
        self.models = models
        self.ml_pipeline = ml_pipeline
        self.rag_engine = FinanceRAGEngine()
        
        # Category aliases for entity extraction across English, Hindi, and Tamil
        self.category_aliases = {
            "Rent": {
                "en": ["rent", "housing", "apartment", "pg", "flat"],
                "hi": ["किराया", "किराए", "मकान किराया", "फ्लैट का किराया", "रूम रेंट", "किराये"],
                "ta": ["வாடகை", "வீட்டு வாடகை", "அடுக்குமாடி வாடகை", "அறை வாடகை", "வாடகைக்கு"]
            },
            "Groceries": {
                "en": ["grocery", "groceries", "supermarket", "ration", "vegetables", "milk", "food items"],
                "hi": ["किराना", "किराने", "राशन", "सब्जी", "सब्जियां", "सब्जियों", "दूध", "घरेलू सामान"],
                "ta": ["மளிகை", "மளிகைப் பொருட்கள்", "காய்கறிகள்", "ரேஷன்", "பால்", "மளிகை கடை"]
            },
            "Eating_Out": {
                "en": ["eating out", "dining", "restaurant", "food", "cafe", "swiggy", "zomato", "takeout", "snacks"],
                "hi": ["बाहर खाना", "होटल", "रेस्टोरेंट", "खाना", "नाश्ता", "स्विगी", "ज़ोमैटो", "डाइनिंग"],
                "ta": ["உணவகம்", "வெளியில் சாப்பிடுவது", "ஹோட்டல்", "உணவு", "சாப்பாடு", "ஸ்விக்கி", "ஜொமாட்டோ"]
            },
            "Transport": {
                "en": ["transport", "commute", "travel", "fuel", "petrol", "diesel", "cab", "uber", "ola", "bus", "metro", "train"],
                "hi": ["परिवहन", "यात्रा", "पेट्रोल", "डीजल", "बस", "मेट्रो", "टैक्सी", "कैब", "किराया भाड़ा", "ट्रेन"],
                "ta": ["போக்குவரத்து", "பயணம்", "பெட்ரோல்", "டீசல்", "பேருந்து", "மெட்ரோ", "வாடகை வண்டி", "ரயில்", "டிராவல்"]
            },
            "Loan_Repayment": {
                "en": ["loan", "emi", "debt", "repayment", "credit card payment"],
                "hi": ["ऋण", "लोन", "ईएमआई", "किस्त", "कर्ज", "क्रेडिट कार्ड"],
                "ta": ["கடன்", "இஎம்ஐ", "மாதத்தவணை", "தவணை", "கடன் திருப்பிச் செலுத்துதல்"]
            },
            "Insurance": {
                "en": ["insurance", "policy", "premium", "lic", "health insurance"],
                "hi": ["बीमा", "पॉलिसी", "प्रीमियम", "एलआईसी", "स्वास्थ्य बीमा"],
                "ta": ["காப்பீடு", "பாலிசி", "பிரீமியம்", "எல்ஐசி", "மருத்துவ காப்பீடு"]
            },
            "Entertainment": {
                "en": ["entertainment", "movie", "cinema", "netflix", "games", "outing", "party", "recreation"],
                "hi": ["मनोरंजन", "फिल्म", "सिनेमा", "मूवी", "नेटफ्लिक्स", "पार्टी", "खेल"],
                "ta": ["பொழுதுபோக்கு", "திரைப்படம்", "சினிமா", "மூவி", "நெட்பிளிக்ஸ்", "பார்ட்டி"]
            },
            "Utilities": {
                "en": ["utility", "utilities", "electricity", "water", "wifi", "internet", "gas", "bill", "phone bill"],
                "hi": ["बिजली", "पानी", "बिल", "इंटरनेट", "वाईफाई", "गैस", "फोन बिल", "उपयोगिता"],
                "ta": ["மின்சாரம்", "தண்ணீர்", "பில்", "இணையம்", "வைஃபை", "எரிவாயு", "தொலைபேசி பில்"]
            },
            "Healthcare": {
                "en": ["health", "healthcare", "medical", "doctor", "medicine", "pharmacy", "hospital", "clinic"],
                "hi": ["स्वास्थ्य", "चिकित्सा", "दवा", "डॉक्टर", "अस्पताल", "मेडिकल", "इलाज"],
                "ta": ["மருத்துவம்", "சுகாதாரம்", "மருந்து", "மருத்துவர்", "மருத்துவமனை", "சிகிச்சை"]
            },
            "Education": {
                "en": ["education", "tuition", "course", "college", "school", "books", "fees", "classes"],
                "hi": ["शिक्षा", "ट्यूशन", "कॉलेज", "स्कूल", "फीस", "किताबें", "कोर्स", "पढ़ाई"],
                "ta": ["கல்வி", "டியூஷன்", "பள்ளி", "கல்லூரி", "கட்டணம்", "புத்தகங்கள்", "படிப்பு"]
            },
            "Miscellaneous": {
                "en": ["misc", "miscellaneous", "other", "shopping", "clothes", "general"],
                "hi": ["विविध", "अन्य", "शॉपिंग", "कपड़े", "साधारण खर्च"],
                "ta": ["இதர", "பிற", "ஷாப்பிங்", "துணிகள்", "பொது செலவுகள்"]
            }
        }

    def detect_language(self, text, requested_lang=None):
        if requested_lang in ["hi", "ta", "en"]:
            return requested_lang
        # Script based detection
        if re.search(r"[\u0B80-\u0BFF]", text):
            return "ta"
        if re.search(r"[\u0900-\u097F]", text):
            return "hi"
        return "en"

    def extract_category(self, text):
        cleaned = text.lower()
        for cat, lang_dict in self.category_aliases.items():
            for lang, aliases in lang_dict.items():
                for alias in aliases:
                    if re.search(r"(?:^|\s|[^\w])" + re.escape(alias) + r"(?:$|\s|[^\w])", cleaned):
                        return cat
        return None

    def extract_timeframe(self, text):
        cleaned = text.lower()
        now = datetime.now()
        
        # Last month
        if any(w in cleaned for w in ["last month", "पिछला महीना", "पिछले महीने", "கடந்த மாதம்"]):
            first_day_current = now.replace(day=1)
            last_day_prev = first_day_current - timedelta(days=1)
            start_date = last_day_prev.replace(day=1)
            return "last_month", start_date, last_day_prev
        # This month
        elif any(w in cleaned for w in ["this month", "current month", "monthly", "इस महीने", "इस माह", "महीने", "இந்த மாதம்", "மாதம்"]):
            start_date = now.replace(day=1)
            return "this_month", start_date, now
        # This year
        elif any(w in cleaned for w in ["this year", "yearly", "इस साल", "वार्षिक", "இந்த ஆண்டு", "ஆண்டு"]):
            start_date = now.replace(month=1, day=1)
            return "this_year", start_date, now
            
        return "all_time", None, None

    def extract_intent(self, text):
        cleaned = text.strip().lower()
        cat = self.extract_category(cleaned)
        
        # 1. Greetings
        if re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b", cleaned) or \
           any(w in cleaned for w in ["नमस्ते", "नमस्कार", "प्रणाम", "हैलो", "வணக்கம்", "ஹலோ"]):
            return "GREETING"
            
        # 2. Help
        if re.search(r"\b(help|what can you do|how to use|commands|options|questions)\b", cleaned) or \
           any(w in cleaned for w in ["मदद", "सहायता", "क्या कर सकते हो", "உதவி", "வழிகாட்டல்"]):
            return "HELP"

        # 3. Financial Advice Safety Guardrail
        if re.search(r"\b(stock|stocks|invest|investing|investment|crypto|cryptocurrency|bitcoin|btc|eth|mutual fund|trade|trading|tax advice|financial advisor|financial advice|advice|tips)\b", cleaned) or \
           any(w in cleaned for w in ["सलाह", "वित्तीय सलाह", "टिप्स", "शेयर", "स्टॉक", "निवेश", "क्रिप्टो", "बिटकॉइन", "टैक्स", "म्यूचुअल फंड", "ஆலோசனை", "நிதி ஆலோசனை", "பங்குகள்", "பங்குச்சந்தை", "முதலீடு", "கிரிப்டோ", "பிட்காயின்", "வரி"]):
            return "ADVICE_DISCLAIMER"

        # 4. Check Balance
        if re.search(r"\b(what('s| is) my balance|how much (money )?do i have|current balance|check balance|net balance|remaining balance|account balance)\b", cleaned) or \
           any(w in cleaned for w in ["बैलेंस", "शेष राशि", "खाते में कितने पैसे", "कितना पैसा बचा", "இருப்பு", "பேலன்ஸ்", "மீதி பணம்", "கணக்கு இருப்பு"]):
            return "CHECK_BALANCE"

        # 5. Financial Health Assessment (ML)
        if re.search(r"\b(financial health|assess my finance|assess financial health|am i saving enough|financial status|health check|savings capacity|ml prediction)\b", cleaned) or \
           any(w in cleaned for w in ["वित्तीय स्वास्थ्य", "आर्थिक स्थिति", "स्वास्थ्य जांच", "நிதி ஆரோக்கியம்", "நிதி நிலை", "ஆரோக்கிய சோதனை"]):
            return "FINANCIAL_HEALTH"

        # 6. Budget Status
        if re.search(r"\b(budget|budget status|how much budget|budget remaining|remaining budget|did i exceed|over budget|budget limit)\b", cleaned) or \
           any(w in cleaned for w in ["बजट", "बजट स्थिति", "बजट बचा", "पार्श्व बजट", "பட்ஜெட்", "பட்ஜெட் நிலை", "மீதமுள்ள பட்ஜெட்"]):
            return "BUDGET_STATUS"

        # 7. Highest / Lowest Expense
        if re.search(r"\b(highest expense|biggest expense|max expense|most expensive|largest expense|maximum spend|what did i spend the most on)\b", cleaned) or \
           any(w in cleaned for w in ["सबसे बड़ा खर्च", "सर्वाधिक खर्च", "सबसे ज्यादा खर्च", "அதிகபட்ச செலவு", "பெரிய செலவு"]):
            return "HIGHEST_EXPENSE"
        if re.search(r"\b(lowest expense|smallest expense|min expense|least expensive|minimum spend)\b", cleaned) or \
           any(w in cleaned for w in ["सबसे कम खर्च", "न्यूनतम खर्च", "छोटा खर्चा", "குறைந்தபட்ச செலவு", "சிறு செலவு"]):
            return "LOWEST_EXPENSE"

        # 8. Top Spending Category
        if re.search(r"\b(top spending|what category do i spend the most|highest spending category|which category|most spent category)\b", cleaned) or \
           any(w in cleaned for w in ["सबसे ज्यादा किस श्रेणी में", "शीर्ष खर्च", "அதிகம் செலவழிக்கும் வகை", "முக்கிய செலவு"]):
            return "TOP_SPENDING_CATEGORY"

        # 9. Recent Transactions
        if re.search(r"\b(recent transaction|recent transactions|last transaction|latest transaction|show transactions|recent activity|past transactions|history)\b", cleaned) or \
           any(w in cleaned for w in ["हाल के लेन-देन", "लेनदेन", "हालिया लेनदेन", "समीபத்திய பரிவர்த்தனைகள்", "வரவு செலவு", "பரிவர்த்தனை"]):
            return "RECENT_TRANSACTIONS"

        # 10. Savings
        if re.search(r"\b(how much (money )?did i save|savings|my savings|saving rate|disposable savings|net savings)\b", cleaned) or \
           any(w in cleaned for w in ["बचत", "कितनी बचत", "सेविंग्स", "சேமிப்பு", "எவ்வளவு சேமித்தேன்", "சேமிப்பு விகிதம்"]):
            return "SAVINGS"

        # 11. Category Expense (If category detected)
        if cat:
            return "CATEGORY_EXPENSE"

        # 12. Monthly Expense
        if any(w in cleaned for w in ["this month", "monthly", "current month", "इस महीने", "इस माह", "महीने का", "இந்த மாதம்", "மாத செலவு"]) and \
           any(w in cleaned for w in ["spend", "spent", "expense", "cost", "खर्च", "व्यय", "செலவு", "செலவழித்த"]):
            return "MONTHLY_EXPENSE"

        # 13. Total Income
        if re.search(r"\b(total income|how much (did i earn|have i earned|income)|my income|earnings|show income|all income)\b", cleaned) or \
           any(w in cleaned for w in ["कुल आय", "कमाई", "कुल कमाई", "மொத்த வருமானம்", "வருமானம்", "வருவாய்"]):
            return "TOTAL_INCOME"

        # 14. Total Expense
        if re.search(r"\b(total expense|total expenses|how much did i spend|how much have i spent|total spending|all expenses|overall spending)\b", cleaned) or \
           any(w in cleaned for w in ["कुल खर्च", "कुल व्यय", "कितना खर्च हुआ", "पूरा खर्चा", "மொத்த செலவு", "செலவுகள்"]):
            return "TOTAL_EXPENSE"

        # Fallbacks
        if any(w in cleaned for w in ["balance", "बैलेंस", "இருப்பு"]):
            return "CHECK_BALANCE"
        if any(w in cleaned for w in ["income", "आय", "कमाई", "வருமானம்"]):
            return "TOTAL_INCOME"
        if any(w in cleaned for w in ["budget", "बजट", "பட்ஜெட்"]):
            return "BUDGET_STATUS"
        if any(w in cleaned for w in ["spend", "expense", "spent", "खर्च", "செலவு"]):
            if any(w in cleaned for w in ["month", "महीना", "மாதம்"]):
                return "MONTHLY_EXPENSE"
            return "TOTAL_EXPENSE"
            
        return "UNKNOWN"

    def process_query(self, user_id, message, requested_lang=None):
        payload = self._process_query_raw(user_id, message, requested_lang)
        if "rag_active" not in payload:
            payload["rag_active"] = True
        if "rag_sources" not in payload:
            payload["rag_sources"] = getattr(self, "_current_citations", [])
        return payload

    def _process_query_raw(self, user_id, message, requested_lang=None):
        lang = self.detect_language(message, requested_lang)
        intent = self.extract_intent(message)
        category = self.extract_category(message)
        timeframe, start_date, end_date = self.extract_timeframe(message)
        
        Transaction = self.models["Transaction"]
        Budget = self.models["Budget"]
        User = self.models["User"]
        
        now = datetime.now()
        current_month = now.month
        current_year = now.year

        # 1. RAG Knowledge & User Evidence Retrieval
        rag_context = self.rag_engine.build_rag_response(
            user_id=user_id,
            db=self.db,
            models=self.models,
            query_text=message,
            intent=intent,
            lang=lang,
            category=category
        )
        domain_chunks = rag_context.get("domain_chunks", [])
        citations = rag_context.get("citations", [])
        self._current_citations = citations
        user_evidence = rag_context.get("user_evidence", {})

        def out(payload):
            if "rag_active" not in payload:
                payload["rag_active"] = True
            if "rag_sources" not in payload:
                payload["rag_sources"] = citations
            return payload

        # Check for direct high-confidence RAG Domain synthesis
        top_chunk = domain_chunks[0] if domain_chunks else None
        if top_chunk and top_chunk["score"] >= 0.07 and (
            intent in ["UNKNOWN", "ADVICE_DISCLAIMER"] or
            any(w in message.lower() for w in [
                "50/30/20", "emergency", "fund", "tax", "80c", "debt", "emi", "benchmark", "save money", "invest",
                "सलाह", "नियम", "इमरजेंसी", "आपातकालीन", "फंड", "बचत", "टैक्स", "ऋण", "कर्ज", "निवेश",
                "ஆலோசனை", "விதி", "சேமிப்பது எப்படி", "சேமிப்பு", "அவசரகால", "வரி", "கடன்", "முதலீடு"
            ])
        ):
            user_inc = user_evidence.get("monthly_income", 65000.0) if user_evidence else 65000.0
            user_outflow = user_evidence.get("total_expense", 0.0) if user_evidence else 0.0
            user_bal = user_evidence.get("balance", 0.0) if user_evidence else 0.0
            cur_month_exp = user_evidence.get("current_month_expense", 35000.0) if user_evidence else 35000.0
            
            rag_title = top_chunk["title"]
            rag_content = top_chunk["content"]
            
            if top_chunk["id"] == "RAG-001":  # 50/30/20 Rule
                needs_tgt = user_inc * 0.50
                wants_tgt = user_inc * 0.30
                save_tgt = user_inc * 0.20
                if lang == "hi":
                    resp = (
                        f"⚡ RAG वित्तीय ज्ञान संवर्धन ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"📊 आपकी वास्तविक आय (₹{user_inc:,.2f}) के आधार पर गणना:\n"
                        f"• अनिवार्य आवश्यकताएं (50%): ₹{needs_tgt:,.2f} तक\n"
                        f"• जीवनशैली एवं इच्छाएं (30%): ₹{wants_tgt:,.2f} तक\n"
                        f"• न्यूनतम लक्ष्य बचत (20%): ₹{save_tgt:,.2f} प्रति माह\n"
                        f"वर्तमान में आपका कुल दर्ज खर्च ₹{user_outflow:,.2f} और शेष राशि ₹{user_bal:,.2f} है।"
                    )
                elif lang == "ta":
                    resp = (
                        f"⚡ RAG நிதி அறிவு ஒருங்கிணைப்பு ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"📊 உங்கள் பதிவு செய்யப்பட்ட வருமானம் (₹{user_inc:,.2f}) அடிப்படையிலான கணக்கீடு:\n"
                        f"• அத்தியாவசியத் தேவைகள் (50%): ₹{needs_tgt:,.2f} வரை\n"
                        f"• விருப்பச் செலவுகள் (30%): ₹{wants_tgt:,.2f} வரை\n"
                        f"• இலக்கு மாதாந்திர சேமிப்பு (20%): ₹{save_tgt:,.2f}\n"
                        f"தற்போது உங்கள் மொத்த பதிவு செலவு ₹{user_outflow:,.2f} மற்றும் இருப்பு ₹{user_bal:,.2f} ஆகும்."
                    )
                else:
                    resp = (
                        f"⚡ RAG Augmented Financial Insight ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"📊 Customized against your recorded monthly income (₹{user_inc:,.2f}):\n"
                        f"• Essential Needs (50% max): ₹{needs_tgt:,.2f}\n"
                        f"• Discretionary Wants (30% max): ₹{wants_tgt:,.2f}\n"
                        f"• Minimum Savings Target (20%): ₹{save_tgt:,.2f}/month\n"
                        f"Your recorded total expenditure is ₹{user_outflow:,.2f} with a net balance of ₹{user_bal:,.2f}."
                    )
                return out({
                    "intent": "RAG_BUDGET_ADVICE",
                    "language": lang,
                    "response": resp,
                    "data": {"income": user_inc, "needs_tgt": needs_tgt, "wants_tgt": wants_tgt, "save_tgt": save_tgt}
                })

            elif top_chunk["id"] == "RAG-002":  # Emergency Fund
                ef_min = cur_month_exp * 3
                ef_max = cur_month_exp * 6
                if lang == "hi":
                    resp = (
                        f"⚡ RAG वित्तीय ज्ञान संवर्धन ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"🛡️ आपके औसत मासिक खर्च (₹{cur_month_exp:,.2f}) के आधार पर आपातकालीन लक्ष्य:\n"
                        f"• न्यूनतम 3-महीने का सुरक्षित बफर: ₹{ef_min:,.2f}\n"
                        f"• अनुशंसित 6-महीने का पूर्ण बफर: ₹{ef_max:,.2f}\n"
                        f"वर्तमान उपलब्ध शेष राशि: ₹{user_bal:,.2f}।"
                    )
                elif lang == "ta":
                    resp = (
                        f"⚡ RAG நிதி அறிவு ஒருங்கிணைப்பு ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"🛡️ உங்கள் மாதாந்திர செலவு (₹{cur_month_exp:,.2f}) அடிப்படையிலான அவசரகால நிதி இலக்கு:\n"
                        f"• குறைந்தபட்ச 3 மாத பாதுகாப்பு நிதி: ₹{ef_min:,.2f}\n"
                        f"• பரிந்துரைக்கப்படும் 6 மாத முழு நிதி: ₹{ef_max:,.2f}\n"
                        f"தற்போதைய இருப்புத் தொகை: ₹{user_bal:,.2f}."
                    )
                else:
                    resp = (
                        f"⚡ RAG Augmented Financial Insight ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"🛡️ Emergency Reserve Milestones (Monthly spending: ₹{cur_month_exp:,.2f}):\n"
                        f"• Baseline 3-Month Emergency Cushion: ₹{ef_min:,.2f}\n"
                        f"• Comprehensive 6-Month Emergency Cushion: ₹{ef_max:,.2f}\n"
                        f"Current Account Liquidity: ₹{user_bal:,.2f}."
                    )
                return out({
                    "intent": "RAG_EMERGENCY_FUND",
                    "language": lang,
                    "response": resp,
                    "data": {"monthly_exp": cur_month_exp, "ef_min": ef_min, "ef_max": ef_max}
                })

            elif top_chunk["id"] == "RAG-004":  # Tax Optimization (Section 80C)
                ins_spent = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                    .filter(Transaction.user_id == user_id, Transaction.category == "Insurance").scalar()
                if lang == "hi":
                    resp = (
                        f"⚡ RAG वित्तीय ज्ञान संवर्धन ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"📑 आपके बहीखाते से कर विश्लेषण:\n"
                        f"• दर्ज बीमा/पॉलिसी व्यय: ₹{ins_spent:,.2f}\n"
                        f"• धारा 80C के तहत अधिकतम छूट सीमा: ₹1,50,000.00 (PPF, ELSS, टर्म प्लान)\n"
                        f"• एनपीएस धारा 80CCD(1B) अतिरिक्त छूट: ₹50,000.00।"
                    )
                elif lang == "ta":
                    resp = (
                        f"⚡ RAG நிதி அறிவு ஒருங்கிணைப்பு ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"📑 உங்கள் லெட்ஜர் தரவு மற்றும் வரி விலக்கு சுருக்கம்:\n"
                        f"• பதிவு செய்யப்பட்ட காப்பீட்டு செலவு: ₹{ins_spent:,.2f}\n"
                        f"• பிரிவு 80C அதிகபட்ச விலக்கு: ₹1,50,000.00 (PPF, ELSS, காப்பீடு)\n"
                        f"• NPS பிரிவு 80CCD(1B) கூடுதல் விலக்கு: ₹50,000.00."
                    )
                else:
                    resp = (
                        f"⚡ RAG Augmented Financial Insight ({rag_title}):\n"
                        f"{rag_content}\n\n"
                        f"📑 Tax Planning Audit against your recorded ledger:\n"
                        f"• Recorded Insurance Outflow: ₹{ins_spent:,.2f}\n"
                        f"• Section 80C Maximum Ceiling: ₹150,000.00 (PPF, ELSS, Term Insurance)\n"
                        f"• Section 80CCD(1B) NPS Additional Deduction: ₹50,000.00."
                    )
                return out({
                    "intent": "RAG_TAX_INSIGHT",
                    "language": lang,
                    "response": resp,
                    "data": {"insurance_spent": ins_spent}
                })
            else:
                if lang == "hi":
                    resp = f"⚡ RAG वित्तीय ज्ञान संवर्धन ({rag_title}):\n{rag_content}"
                elif lang == "ta":
                    resp = f"⚡ RAG நிதி அறிவு ஒருங்கிணைப்பு ({rag_title}):\n{rag_content}"
                else:
                    resp = f"⚡ RAG Augmented Financial Insight ({rag_title}):\n{rag_content}"
                return out({
                    "intent": "RAG_KNOWLEDGE",
                    "language": lang,
                    "response": resp,
                    "data": {"chunk_id": top_chunk["id"]}
                })

        # 1. GREETING
        if intent == "GREETING":
            if lang == "hi":
                resp = "नमस्ते! मैं आपका FinAI वित्तीय सहायक हूँ। मैं आपके खर्चों का विश्लेषण करने, आपके बैलेंस की जांच करने, मासिक बजट की निगरानी करने और आपके वास्तविक वित्तीय लेन-देन के आधार पर AI वित्तीय स्वास्थ्य सलाह देने में मदद कर सकता हूँ। मैं आज आपकी क्या सहायता करूँ?"
            elif lang == "ta":
                resp = "வணக்கம்! நான் உங்கள் FinAI நிதி உதவியாளர். உங்கள் செலவுகளைப் பகுப்பாய்வு செய்யவும், உங்கள் கணக்கு இருப்பைச் சரிபார்க்கவும், மாதாந்திர பட்ஜெட்டைக் கண்காணிக்கவும், உங்கள் உண்மையான நிதிப் பரிவர்த்தனைகளின் அடிப்படையில் AI நிதி ஆரோக்கிய ஆலோசனைகளை வழங்கவும் என்னால் முடியும். இன்று உங்களுக்கு நான் எவ்வாறு உதவட்டும்?"
            else:
                resp = "Hello! I am your FinAI Finance Assistant. I can help analyze your expenses, check your balance, monitor your monthly budget, and provide AI financial health insights based on your real transactions. How can I assist you today?"
            return {"intent": intent, "language": lang, "response": resp, "data": {}}

        # 2. HELP
        if intent == "HELP":
            if lang == "hi":
                resp = "यहाँ कुछ प्रश्न हैं जो आप मुझसे पूछ सकते हैं:\n" \
                       "• 'मेरा वर्तमान बैलेंस क्या है?'\n" \
                       "• 'इस महीने मैंने कितना खर्च किया?'\n" \
                       "• 'किराने पर कितना खर्च हुआ?'\n" \
                       "• 'मेरा बजट स्टेटस क्या है?'\n" \
                       "• 'सबसे बड़ा खर्चा कौन सा था?'\n" \
                       "• 'हाल के लेन-देन दिखाओ।'\n" \
                       "• 'मेरे वित्तीय स्वास्थ्य का आकलन करें।'"
            elif lang == "ta":
                resp = "நீங்கள் என்னிடம் கேட்கக்கூடிய சில கேள்விகள் இங்கே:\n" \
                       "• 'என் தற்போதைய இருப்பு என்ன?'\n" \
                       "• 'இந்த மாதம் நான் எவ்வளவு செலவு செய்தேன்?'\n" \
                       "• 'மளிகைப் பொருட்களுக்கு எவ்வளவு செலவானது?'\n" \
                       "• 'என் பட்ஜெட் நிலை என்ன?'\n" \
                       "• 'மிகப் பெரிய செலவு எது?'\n" \
                       "• 'சமீபத்திய பரிவர்த்தனைகளைக் காட்டு.'\n" \
                       "• 'எனது நிதி ஆரோக்கியத்தை மதிப்பிடுங்கள்.'"
            else:
                resp = "Here are some questions you can ask me:\n" \
                       "• 'What is my current balance?'\n" \
                       "• 'How much did I spend this month?'\n" \
                       "• 'How much did I spend on groceries?'\n" \
                       "• 'What is my budget status?'\n" \
                       "• 'What was my highest expense?'\n" \
                       "• 'Show my recent transactions.'\n" \
                       "• 'Assess my financial health.'"
            return {"intent": intent, "language": lang, "response": resp, "data": {}}

        # 3. SAFETY DISCLAIMER
        if intent == "ADVICE_DISCLAIMER":
            if lang == "hi":
                resp = "मैं सामान्य सूचनात्मक मार्गदर्शन प्रदान कर सकता हूँ और इस एप्लिकेशन में दर्ज वित्तीय डेटा का विश्लेषण कर सकता हूँ, लेकिन यह पेशेवर वित्तीय सलाह नहीं है। विशिष्ट निवेश, स्टॉक ट्रेडिंग या टैक्स सलाह के लिए कृपया प्रमाणित वित्तीय योजनाकार से संपर्क करें।"
            elif lang == "ta":
                resp = "நான் பொதுவான தகவல்களை வழங்கி உங்கள் நிதித் தரவை பகுப்பாய்வு செய்ய முடியும், ஆனால் இது தொழில்முறை நிதி ஆலோசனை அல்ல. குறிப்பிட்ட முதலீடு, பங்குச் சந்தை அல்லது வரி திட்டமிடலுக்கு சான்றளிக்கப்பட்ட நிதி ஆலோசகரை அணுகவும்."
            else:
                resp = "I can provide general informational guidance and analyze the financial data recorded in this application, but this is not professional financial advice. For personalized investment, stock picking, or tax filing advice, please consult a certified financial planner."
            return {"intent": intent, "language": lang, "response": resp, "data": {}}

        # 4. CHECK BALANCE
        if intent == "CHECK_BALANCE":
            total_income = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'income').scalar()
            total_expense = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense').scalar()
            balance = total_income - total_expense
            
            if lang == "hi":
                resp = f"आपकी वर्तमान शेष राशि (बैलेंस) ₹{balance:,.2f} है।\n(कुल आय: ₹{total_income:,.2f} | कुल खर्च: ₹{total_expense:,.2f})"
            elif lang == "ta":
                resp = f"உங்கள் தற்போதைய இருப்புத் தொகை ₹{balance:,.2f}.\n(மொத்த வரவு: ₹{total_income:,.2f} | மொத்த செலவு: ₹{total_expense:,.2f})"
            else:
                resp = f"Your current balance is ₹{balance:,.2f}.\n(Total Income: ₹{total_income:,.2f} | Total Expenses: ₹{total_expense:,.2f})"
                
            return {"intent": intent, "language": lang, "response": resp, "data": {"balance": balance, "total_income": total_income, "total_expense": total_expense}}

        # 5. TOTAL INCOME
        if intent == "TOTAL_INCOME":
            q = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'income')
            if start_date:
                q = q.filter(Transaction.date >= start_date.strftime("%Y-%m-%d"))
            if end_date:
                q = q.filter(Transaction.date <= end_date.strftime("%Y-%m-%d"))
            total_income = q.scalar()
            
            if lang == "hi":
                resp = f"आपकी दर्ज की गई कुल आय ₹{total_income:,.2f} है।"
            elif lang == "ta":
                resp = f"பதிவு செய்யப்பட்ட உங்கள் மொத்த வருமானம் ₹{total_income:,.2f} ஆகும்."
            else:
                resp = f"Your total recorded income is ₹{total_income:,.2f}."
            return {"intent": intent, "language": lang, "response": resp, "data": {"total_income": total_income}}

        # 6. TOTAL EXPENSE
        if intent == "TOTAL_EXPENSE":
            total_expense = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense').scalar()
            if lang == "hi":
                resp = f"आपका कुल खर्च ₹{total_expense:,.2f} है।"
            elif lang == "ta":
                resp = f"உங்கள் மொத்த செலவுத் தொகை ₹{total_expense:,.2f} ஆகும்."
            else:
                resp = f"Your overall total expenses amount to ₹{total_expense:,.2f}."
            return {"intent": intent, "language": lang, "response": resp, "data": {"total_expense": total_expense}}

        # 7. MONTHLY EXPENSE
        if intent == "MONTHLY_EXPENSE":
            month_start = now.replace(day=1).strftime("%Y-%m-%d")
            monthly_expense = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense', Transaction.date >= month_start).scalar()
            if lang == "hi":
                resp = f"आपने इस महीने ({now.strftime('%B %Y')}) में अब तक ₹{monthly_expense:,.2f} खर्च किए हैं।"
            elif lang == "ta":
                resp = f"இந்த மாதத்தில் ({now.strftime('%B %Y')}) இதுவரை நீங்கள் ₹{monthly_expense:,.2f} செலவழித்துள்ளீர்கள்."
            else:
                resp = f"You have spent ₹{monthly_expense:,.2f} so far this month ({now.strftime('%B %Y')})."
            return {"intent": intent, "language": lang, "response": resp, "data": {"monthly_expense": monthly_expense, "month": now.strftime('%B %Y')}}

        # 8. CATEGORY EXPENSE
        if intent == "CATEGORY_EXPENSE":
            if not category:
                top_cat = self.db.session.query(Transaction.category, func.sum(Transaction.amount).label("total"))\
                    .filter(Transaction.user_id == user_id, Transaction.type == 'expense')\
                    .group_by(Transaction.category).order_by(func.sum(Transaction.amount).desc()).first()
                if top_cat:
                    cat_name = top_cat[0]
                    cat_amt = top_cat[1]
                    if lang == "hi":
                        resp = f"आपकी सबसे ज्यादा खर्च वाली श्रेणी {cat_name} है, जिसमें कुल ₹{cat_amt:,.2f} खर्च हुए हैं।"
                    elif lang == "ta":
                        resp = f"உங்கள் அதிக செலவுப் பிரிவு {cat_name} ஆகும், இதில் மொத்தம் ₹{cat_amt:,.2f} செலவாகியுள்ளது."
                    else:
                        resp = f"Your highest expense category is {cat_name} with ₹{cat_amt:,.2f} in total spending."
                    return {"intent": intent, "language": lang, "response": resp, "data": {"category": cat_name, "amount": cat_amt}}
                else:
                    empty_msg = "कोई खर्च दर्ज नहीं है।" if lang == "hi" else ("செலவுகள் எதுவும் பதிவு செய்யப்படவில்லை." if lang == "ta" else "You don't have any recorded expenses yet.")
                    return {"intent": intent, "language": lang, "response": empty_msg, "data": {}}
            
            q = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense', Transaction.category == category)
            if start_date:
                q = q.filter(Transaction.date >= start_date.strftime("%Y-%m-%d"))
            if end_date:
                q = q.filter(Transaction.date <= end_date.strftime("%Y-%m-%d"))
            amount = q.scalar()
            
            if lang == "hi":
                resp = f"आपने {category} पर ₹{amount:,.2f} खर्च किए हैं।"
            elif lang == "ta":
                resp = f"நீங்கள் {category} பிரிவில் ₹{amount:,.2f} செலவிட்டுள்ளீர்கள்."
            else:
                resp = f"You spent ₹{amount:,.2f} on {category}."
            return {"intent": intent, "language": lang, "response": resp, "data": {"category": category, "amount": amount}}

        # 9. BUDGET STATUS
        if intent == "BUDGET_STATUS":
            budget_obj = Budget.query.filter_by(user_id=user_id, month=current_month, year=current_year).first()
            budget_amount = budget_obj.amount if budget_obj else 0.0
            
            month_start = now.replace(day=1).strftime("%Y-%m-%d")
            spent = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense', Transaction.date >= month_start).scalar()
            
            if budget_amount == 0:
                if lang == "hi":
                    resp = f"आपने इस महीने ₹{spent:,.2f} खर्च किए हैं, लेकिन अभी तक कोई बजट सीमा निर्धारित नहीं की है।"
                elif lang == "ta":
                    resp = f"இந்த மாதம் நீங்கள் ₹{spent:,.2f} செலவிட்டுள்ளீர்கள், ஆனால் இன்னும் பட்ஜெட் வரம்பு அமைக்கவில்லை."
                else:
                    resp = f"You have spent ₹{spent:,.2f} this month, but you haven't set a budget limit yet."
                return {"intent": intent, "language": lang, "response": resp, "data": {"spent": spent, "budget": 0}}
                
            remaining = budget_amount - spent
            pct_used = (spent / budget_amount) * 100 if budget_amount > 0 else 0
            
            if remaining < 0:
                if lang == "hi":
                    resp = f"चेतावनी: आप अपने मासिक बजट से ₹{abs(remaining):,.2f} अधिक खर्च कर चुके हैं ({pct_used:.1f}% उपयोग हुआ)।"
                elif lang == "ta":
                    resp = f"எச்சரிக்கை: உங்கள் மாதாந்திர பட்ஜெட்டை விட ₹{abs(remaining):,.2f} அதிகமாகச் செலவழித்துள்ளீர்கள் ({pct_used:.1f}% பயன்படுத்தப்பட்டது)."
                else:
                    resp = f"Alert: You have exceeded your monthly budget by ₹{abs(remaining):,.2f} ({pct_used:.1f}% used)."
            elif pct_used >= 85:
                if lang == "hi":
                    resp = f"सावधान: आपने अपने बजट का {pct_used:.1f}% उपयोग कर लिया है! केवल ₹{remaining:,.2f} शेष है।"
                elif lang == "ta":
                    resp = f"எச்சரிக்கை: பட்ஜெட்டில் {pct_used:.1f}% செலவாகிவிட்டது! வெறும் ₹{remaining:,.2f} மட்டுமே மீதமுள்ளது."
                else:
                    resp = f"Warning: You have used {pct_used:.1f}% of your budget! Only ₹{remaining:,.2f} remains."
            else:
                if lang == "hi":
                    resp = f"शानदार! आपके ₹{budget_amount:,.2f} के बजट में से ₹{remaining:,.2f} शेष है ({pct_used:.1f}% उपयोग हुआ)।"
                elif lang == "ta":
                    resp = f"நன்று! உங்கள் ₹{budget_amount:,.2f} பட்ஜெட்டில் ₹{remaining:,.2f} மீதமுள்ளது ({pct_used:.1f}% பயன்படுத்தப்பட்டது)."
                else:
                    resp = f"Good job! You have ₹{remaining:,.2f} remaining out of your ₹{budget_amount:,.2f} budget ({pct_used:.1f}% used)."
                    
            return {"intent": intent, "language": lang, "response": resp, "data": {"budget": budget_amount, "spent": spent, "remaining": remaining, "percent_used": round(pct_used, 1)}}

        # 10. HIGHEST EXPENSE
        if intent == "HIGHEST_EXPENSE":
            top_tx = Transaction.query.filter_by(user_id=user_id, type='expense')\
                .order_by(Transaction.amount.desc()).first()
            if top_tx:
                if lang == "hi":
                    resp = f"आपका सबसे बड़ा खर्च {top_tx.date} को {top_tx.category} ('{top_tx.description}') पर ₹{top_tx.amount:,.2f} था।"
                elif lang == "ta":
                    resp = f"உங்கள் மிகப்பெரிய செலவு {top_tx.date} அன்று {top_tx.category} ('{top_tx.description}') பிரிவில் ₹{top_tx.amount:,.2f} ஆகும்."
                else:
                    resp = f"Your highest recorded expense is ₹{top_tx.amount:,.2f} on {top_tx.category} ('{top_tx.description}') on {top_tx.date}."
                return {"intent": intent, "language": lang, "response": resp, "data": {"amount": top_tx.amount, "category": top_tx.category}}
            empty_msg = "कोई खर्च नहीं मिला।" if lang == "hi" else ("செலவு எதுவும் இல்லை." if lang == "ta" else "No expense records found yet.")
            return {"intent": intent, "language": lang, "response": empty_msg, "data": {}}

        # 11. LOWEST EXPENSE
        if intent == "LOWEST_EXPENSE":
            low_tx = Transaction.query.filter_by(user_id=user_id, type='expense')\
                .order_by(Transaction.amount.asc()).first()
            if low_tx:
                if lang == "hi":
                    resp = f"आपका सबसे छोटा खर्च {low_tx.date} को {low_tx.category} ('{low_tx.description}') पर ₹{low_tx.amount:,.2f} था।"
                elif lang == "ta":
                    resp = f"உங்கள் மிகக் குறைந்த செலவு {low_tx.date} அன்று {low_tx.category} ('{low_tx.description}') பிரிவில் ₹{low_tx.amount:,.2f} ஆகும்."
                else:
                    resp = f"Your smallest recorded expense is ₹{low_tx.amount:,.2f} for {low_tx.category} ('{low_tx.description}') on {low_tx.date}."
                return {"intent": intent, "language": lang, "response": resp, "data": {"amount": low_tx.amount, "category": low_tx.category}}
            empty_msg = "कोई खर्च नहीं मिला।" if lang == "hi" else ("செலவு எதுவும் இல்லை." if lang == "ta" else "No expense records found yet.")
            return {"intent": intent, "language": lang, "response": empty_msg, "data": {}}

        # 12. RECENT TRANSACTIONS
        if intent == "RECENT_TRANSACTIONS":
            recent = Transaction.query.filter_by(user_id=user_id)\
                .order_by(Transaction.date.desc(), Transaction.id.desc()).limit(5).all()
            if not recent:
                empty_msg = "कोई हालिया लेन-देन नहीं है।" if lang == "hi" else ("சமீபத்திய பரிவர்த்தனைகள் எதுவும் இல்லை." if lang == "ta" else "You don't have any recent transactions recorded.")
                return {"intent": intent, "language": lang, "response": empty_msg, "data": {"transactions": []}}
                
            header = "यहाँ आपके हाल के 5 लेन-देन हैं:" if lang == "hi" else ("உங்கள் சமீபத்திய 5 பரிவர்த்தனைகள் இங்கே:" if lang == "ta" else "Here are your 5 most recent transactions:")
            lines = [header]
            for t in recent:
                sign = "+" if t.type == "income" else "-"
                lines.append(f"• {t.date} | {t.category}: {sign}₹{t.amount:,.2f} ({t.description})")
            return {"intent": intent, "language": lang, "response": "\n".join(lines), "data": {}}

        # 13. SAVINGS
        if intent == "SAVINGS":
            total_income = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'income').scalar()
            total_expense = self.db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense').scalar()
            savings = total_income - total_expense
            rate = (savings / total_income * 100) if total_income > 0 else 0.0
            
            if lang == "hi":
                resp = f"आपकी अब तक की शुद्ध बचत ₹{savings:,.2f} है, जो {rate:.1f}% बचत दर दर्शाती है।"
            elif lang == "ta":
                resp = f"இன்று வரை உங்கள் நிகர சேமிப்பு ₹{savings:,.2f} ஆகும் ({rate:.1f}% சேமிப்பு விகிதம்)."
            else:
                resp = f"Your net savings to date is ₹{savings:,.2f}, representing a {rate:.1f}% savings rate."
            return {"intent": intent, "language": lang, "response": resp, "data": {"savings": savings, "savings_rate": round(rate, 1)}}

        # 14. TOP SPENDING CATEGORY
        if intent == "TOP_SPENDING_CATEGORY":
            top_cat = self.db.session.query(Transaction.category, func.sum(Transaction.amount).label("total"))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense')\
                .group_by(Transaction.category).order_by(func.sum(Transaction.amount).desc()).first()
            if top_cat:
                if lang == "hi":
                    resp = f"आपकी नंबर 1 खर्च श्रेणी {top_cat[0]} है, जिसमें कुल ₹{top_cat[1]:,.2f} खर्च हुए हैं।"
                elif lang == "ta":
                    resp = f"உங்கள் முதன்மை செலவுப் பிரிவு {top_cat[0]} ஆகும், இதில் மொத்தம் ₹{top_cat[1]:,.2f} செலவிடப்பட்டுள்ளது."
                else:
                    resp = f"Your #1 spending category is {top_cat[0]} with a total expenditure of ₹{top_cat[1]:,.2f}."
                return {"intent": intent, "language": lang, "response": resp, "data": {"category": top_cat[0], "total": top_cat[1]}}
            empty_msg = "कोई खर्च नहीं मिला।" if lang == "hi" else ("செலவு எதுவும் இல்லை." if lang == "ta" else "You have no expenses recorded yet.")
            return {"intent": intent, "language": lang, "response": empty_msg, "data": {}}

        # 15. FINANCIAL HEALTH ASSESSMENT (ML Powered)
        if intent == "FINANCIAL_HEALTH":
            user = User.query.get(user_id)
            if not user or not self.ml_pipeline:
                return {"intent": intent, "language": lang, "response": "AI Model not loaded.", "data": {}}
                
            month_start = now.replace(day=1).strftime("%Y-%m-%d")
            monthly_income = user.monthly_income or 50000.0
            
            cat_sums = dict(self.db.session.query(Transaction.category, func.sum(Transaction.amount))\
                .filter(Transaction.user_id == user_id, Transaction.type == 'expense', Transaction.date >= month_start)\
                .group_by(Transaction.category).all())
                
            input_dict = {
                "Occupation": user.occupation or "Professional",
                "City_Tier": user.city_tier or "Tier_2",
                "Income": float(monthly_income),
                "Age": int(user.age or 30),
                "Dependents": int(user.dependents or 1),
                "Rent": float(cat_sums.get("Rent", monthly_income * 0.2)),
                "Loan_Repayment": float(cat_sums.get("Loan_Repayment", 0.0)),
                "Insurance": float(cat_sums.get("Insurance", monthly_income * 0.03)),
                "Groceries": float(cat_sums.get("Groceries", monthly_income * 0.12)),
                "Transport": float(cat_sums.get("Transport", monthly_income * 0.06)),
                "Eating_Out": float(cat_sums.get("Eating_Out", monthly_income * 0.04)),
                "Entertainment": float(cat_sums.get("Entertainment", monthly_income * 0.03)),
                "Utilities": float(cat_sums.get("Utilities", monthly_income * 0.05)),
                "Healthcare": float(cat_sums.get("Healthcare", monthly_income * 0.03)),
                "Education": float(cat_sums.get("Education", 0.0)),
                "Miscellaneous": float(cat_sums.get("Miscellaneous", monthly_income * 0.02))
            }
            
            import pandas as pd
            input_df = pd.DataFrame([input_dict])
            pred_class = int(self.ml_pipeline.predict(input_df)[0])
            prob = float(self.ml_pipeline.predict_proba(input_df)[0][1])
            
            total_spent = sum(cat_sums.values())
            savings_pct = ((monthly_income - total_spent) / monthly_income) * 100 if monthly_income > 0 else 0
            
            if pred_class == 1:
                if lang == "hi":
                    status_text = "स्वस्थ बचतकर्ता (कम जोखिम)"
                    advice = "आपकी खर्च करने की आदतें एक मजबूत बचत बफर (>=20% बचत क्षमता) दर्शाती हैं। अपना आपातकालीन फंड बनाए रखें और दीर्घकालिक निवेश पर ध्यान दें।"
                elif lang == "ta":
                    status_text = "ஆரோக்கியமான சேமிப்பாளர் (குறைந்த ஆபத்து)"
                    advice = "உங்கள் செலவுப் பழக்கம் வலுவான சேமிப்பு இருப்பைக் காட்டுகிறது (>=20%). அவசரக்கால நிதியைப் பராமரித்து நீண்ட கால முதலீடுகளைத் திட்டமிடுங்கள்."
                else:
                    status_text = "Healthy Saver"
                    advice = "Your spending habits show a strong disposable savings buffer (>=20% savings capacity). Maintain your emergency fund and prioritize disciplined investments."
            else:
                if lang == "hi":
                    status_text = "वित्तीय रूप से संकुचित / जोखिम में"
                    advice = "आपकी बचत दर 20% से कम है। भोजन, मनोरंजन और अतिरिक्त खर्चों को कम करके बचत बढ़ाने पर विचार करें।"
                elif lang == "ta":
                    status_text = "நிதி நெருக்கடி / எச்சரிக்கை"
                    advice = "உங்கள் சேமிப்பு அளவு 20%-க்கும் குறைவாக உள்ளது. தேவையற்ற செலவுகளைக் குறைத்து சேமிப்பை உயர்த்தவும்."
                else:
                    status_text = "Financially Constrained / At-Risk"
                    advice = "Your current spending leaves less than 20% savings buffer. Consider optimizing discretionary expenses in Dining Out and Entertainment to boost surplus."
                    
            if lang == "hi":
                resp = f"AI वित्तीय स्वास्थ्य मूल्यांकन:\n• स्थिति: {status_text} (AI विश्वास: {prob*100:.1f}%)\n• अनुमानित मासिक बचत दर: {savings_pct:.1f}%\n• सलाह: {advice}"
            elif lang == "ta":
                resp = f"AI நிதி ஆரோக்கிய மதிப்பீடு:\n• நிலை: {status_text} (AI நம்பிக்கை: {prob*100:.1f}%)\n• உத்தேச மாதாந்திர சேமிப்பு விகிதம்: {savings_pct:.1f}%\n• ஆலோசனை: {advice}"
            else:
                resp = f"AI Financial Health Evaluation:\n• Status: {status_text} (AI Confidence: {prob*100:.1f}%)\n• Estimated Monthly Savings Rate: {savings_pct:.1f}%\n• Recommendation: {advice}"
                
            return {"intent": intent, "language": lang, "response": resp, "data": {"status": status_text, "confidence": round(prob * 100, 1), "savings_rate": round(savings_pct, 1)}}

        # 16. UNKNOWN INTENT FALLBACK
        if lang == "hi":
            fallback = "मुझे ठीक से समझ नहीं आया। आप मुझसे बैलेंस, कुल खर्च, मासिक खर्च, बजट स्थिति, हाल के लेन-देन या वित्तीय स्वास्थ्य के बारे में पूछ सकते हैं।"
        elif lang == "ta":
            fallback = "மன்னிக்கவும், எனக்கு புரியவில்லை. கணக்கு இருப்பு, மாதாந்திர செலவு, பட்ஜெட் நிலை அல்லது நிதி ஆரோக்கியம் பற்றி நீங்கள் என்னிடம் கேட்கலாம்."
        else:
            fallback = "I didn't quite catch that. You can ask me about your balance, total spending, monthly expenses, category breakdown, budget status, recent transactions, or ask me to assess your financial health."
            
        return {"intent": "UNKNOWN", "language": lang, "response": fallback, "data": {}}
