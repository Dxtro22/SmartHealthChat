import * as dotenv from 'dotenv';
dotenv.config();

import http from 'http';
import url from 'url';
import Groq from 'groq-sdk';

import { Pinecone } from '@pinecone-database/pinecone';

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const History = [];

async function getEmbedding(text) {
    const response = await fetch(
        "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction",
        {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${process.env.HF_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ inputs: text, options: { wait_for_model: true } })
        }
    );
    const data = await response.json();
    return Array.isArray(data[0]) ? data[0] : data;
}

async function transformQuery(question) {
    try {
        const response = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            messages: [
                {
                    role: "system",
                    content: `You are a query classification and rewriting expert.
                    Analyze the user's input and follow these STRICT CATEGORIZATION RULES:
                    1. If the input is a casual greeting (e.g., "hi", "hello", "good morning"), output EXACTLY: "GREETING".
                    2. If the input is gibberish or completely unreadable, output EXACTLY: "UNCLEAR".
                    3. For all other inputs, rephrase the health question into a standalone question IN ENGLISH for database searching.
                    Only output the rewritten question, "GREETING", or "UNCLEAR".`
                },
                { role: "user", content: question }
            ],
            max_tokens: 100
        });
        return response.choices[0]?.message?.content || "UNCLEAR";
    } catch (error) {
        console.error("Translation Error:", error.message);
        return "UNCLEAR";
    }
}

async function chatting(question, selectedLang) {
    const queries = await transformQuery(question);
    const isUnclear = queries.toUpperCase().includes("UNCLEAR");
    const isGreeting = queries.toUpperCase().includes("GREETING");

    if (isUnclear) {
        return "I'm sorry, I didn't quite catch that. Could you please describe your health concern clearly?";
    }

    let context = "";

    if (!isGreeting) {
        try {
            const queryVector = await getEmbedding(queries);
            const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
            const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
            const searchResults = await pineconeIndex.query({
                topK: 10,
                vector: queryVector,
                includeMetadata: true,
            });
            context = searchResults.matches.map(match => match.metadata.text).join("\n\n---\n\n");
        } catch (dbError) {
            console.error("Database search failed:", dbError.message);
        }
    }

    History.push({ role: "user", content: question });

    try {
        const messages = [
            {
                role: "system",
                content: `You are 'SmartHealthChat', a warm, empathetic, and thorough virtual health advisor.

**CRITICAL LANGUAGE INSTRUCTION - THIS OVERRIDES EVERYTHING:**
The user has selected language code: '${selectedLang}'
You MUST respond in ONLY this language, no matter what language the user typed in:
- en-IN → respond in ENGLISH only
- hi-IN → respond in HINDI (हिंदी) only
- ta-IN → respond in TAMIL (தமிழ்) only
- te-IN → respond in TELUGU (తెలుగు) only
- mr-IN → respond in MARATHI (मराठी) only
- bn-IN → respond in BENGALI (বাংলা) only
- gu-IN → respond in GUJARATI (ગુજરાતી) only
- kn-IN → respond in KANNADA (ಕನ್ನಡ) only
- ml-IN → respond in MALAYALAM (മലയാളം) only
DO NOT use any other language. The selected code is '${selectedLang}' — use ONLY that language.

**Your Guiding Rules:**
1. **Greetings:** If the user just says hello, greet them back warmly.
2. **Safety First:** If symptoms indicate an emergency (chest pain, severe bleeding), advise immediate emergency care.
3. **Domain Restriction:** You are a HEALTH chatbot only. For non-health questions, politely say you only handle health topics.
4. **Differential Diagnosis:** If symptoms are general, ask follow-up questions to narrow down.
5. **Empathy & Solutions:** Validate their feelings and offer safe home remedies before advising a doctor visit.
6. **Disclaimer:** ALWAYS end medical advice with: "Just a quick reminder: I'm an AI assistant and not a medical professional. For personal medical advice, please consult a doctor."

Context from WHO medical database:
${context}`
            },
            ...History
        ];

        const response = await groq.chat.completions.create({
            model: "llama-3.3-70b-versatile",
            messages: messages,
            max_tokens: 1000
        });

        const answer = response.choices[0]?.message?.content || "Sorry, I could not generate a response.";
        History.push({ role: "assistant", content: answer });
        return answer;

    } catch (finalError) {
        console.error("🔥 API ERROR:", finalError.message);
        History.pop();
        return "[SYSTEM] An unexpected error occurred. Please try again.";
    }
}

const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json; charset=utf-8');

    const parsedUrl = url.parse(req.url, true);

    if (parsedUrl.pathname === '/ask' && parsedUrl.query.q) {
        const question = parsedUrl.query.q;
        const selectedLang = parsedUrl.query.lang || 'en-IN';
        console.log(`\n=> Received: "${question}" | Lang: ${selectedLang}`);

        try {
            const answer = await chatting(question, selectedLang);
            res.statusCode = 200;
            res.end(JSON.stringify({ answer }));
        } catch (error) {
            console.error("Server Crash:", error);
            res.statusCode = 500;
            res.end(JSON.stringify({ error: "Sorry, something went wrong." }));
        }
    } else {
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "Not found." }));
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`SmartHealthChat server running at http://localhost:${PORT}`);
});