import * as dotenv from 'dotenv';
dotenv.config();

import http from 'http';
import url from 'url';

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({});
const History = []; 

// 1. Query Rewriter, Gibberish Filter, & Greeting Detector
async function transformQuery(question){
    try {
        const tempHistory = [...History, { role: 'user', parts: [{ text: question }] }];

        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: tempHistory,
            config: {
              systemInstruction: `You are a query classification and rewriting expert.
              Analyze the user's input and follow these STRICT CATEGORIZATION RULES:
              
              1. If the input is a casual greeting (e.g., "hi", "hello", "good morning", "how are you"), output EXACTLY the word: "GREETING".
              2. If the input is gibberish, random letters (e.g., 'asdfg'), or completely unreadable, output EXACTLY the word: "UNCLEAR".
              3. For all other inputs, rephrase the health question into a standalone question **IN ENGLISH** for database searching.
              
              Only output the rewritten question, "GREETING", or "UNCLEAR".`,
            },
        });
         
        if (!response.text) return "UNCLEAR";
        return response.text;

    } catch (error) {
        console.error("Translation Error:", error.message);
        return "UNCLEAR"; 
    }
}

// 2. Main Chat Function
async function chatting(question, selectedLang) {
    const queries = await transformQuery(question);
    
    const isUnclear = queries.toUpperCase().includes("UNCLEAR");
    const isGreeting = queries.toUpperCase().includes("GREETING");
    
    // --- BUCKET 1: GIBBERISH (Short-Circuit) ---
    if (isUnclear) {
        return "I'm sorry, I didn't quite catch that. Could you please describe your health concern or symptoms clearly?";
    }
    
    let context = ""; 
    
    // --- BUCKET 2: MEDICAL QUERY (Database Search) ---
    // We ONLY search the database if it is NOT a greeting. 
    // This saves time and API costs!
    if (!isGreeting) {
        try {
            const embeddings = new GoogleGenerativeAIEmbeddings({
                apiKey: process.env.GEMINI_API_KEY,
                model: 'gemini-embedding-001', 
                taskType: "RETRIEVAL_QUERY" 
            });
            const queryVector = await embeddings.embedQuery(queries); 
         
            const pinecone = new Pinecone();
            const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
            const searchResults = await pineconeIndex.query({
                topK: 10,
                vector: queryVector,
                includeMetadata: true,
            });

            context = searchResults.matches
                               .map(match => match.metadata.text)
                               .join("\n\n---\n\n");
        } catch (dbError) {
            console.error("Database search failed:", dbError.message);
        }
    }

    // --- Proceed to Main AI Doctor ---
    History.push({ role: 'user', parts: [{ text: question }] });
    
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: History,
            config: {
                systemInstruction: `You are 'SmartHealthChat', a warm, empathetic, and thorough virtual health advisor. 

**CRITICAL LANGUAGE OVERRIDE:** The user has selected the language code '${selectedLang}'. You MUST write your final response entirely in the language that matches this code. Ignore the language the user typed in; ALWAYS reply in the language of '${selectedLang}'.

**Your Guiding Rules:**
1. **Greetings:** If the user just says hello or greets you, greet them back warmly and ask how you can help with their health today.
2. **Safety First:** If symptoms indicate an emergency (chest pain, severe bleeding), advise immediate emergency care.
3. **Domain Restriction:** You are a HEALTH chatbot only. If a user asks non-health questions (like "Who is Virat Kohli?"), politely say: "I am a specialized health assistant and only have information regarding health and medical topics. How can I help you feel better today?"
4. **Differential Diagnosis:** If symptoms are general (fever, headache), do NOT jump to a disease. Ask follow-up questions to narrow it down.
5. **Empathy & Solutions:** Validate their feelings and offer safe home remedies from your context before advising a doctor visit.
6. **Disclaimer:** ALWAYS end medical advice with: "Just a quick reminder: I'm an AI assistant and not a medical professional. For personal medical advice, please consult a doctor."

Here is your medical knowledge base to use for this response (this will be empty if the user is just saying hello):
Context: ${context}`,
            },
        });

        History.push({ role: 'model', parts: [{ text: response.text }] });
        return response.text;

    } catch (finalError) {
        // INTELLIGENT ROUTING & ROLLBACK
        const errorMessage = finalError.message ? finalError.message.toLowerCase() : String(finalError).toLowerCase();
        console.error("🔥 CRITICAL API ERROR:", errorMessage);
        
        History.pop(); 
        
        if (errorMessage.includes("429") || errorMessage.includes("quota")) {
            return "[SYSTEM] The AI is currently rate-limited (too many requests). Please wait 60 seconds and try again.";
        } else if (errorMessage.includes("timeout") || errorMessage.includes("fetch")) {
            return "[SYSTEM] Network connection to Google servers failed. Please check your internet connection.";
        } else {
            return `[SYSTEM] An unexpected backend error occurred. Please try again.`;
        }
    }
}

// 3. Server Setup
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
            res.end(JSON.stringify({ answer: answer }));
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