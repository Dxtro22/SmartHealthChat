import * as dotenv from 'dotenv';
dotenv.config();

import http from 'http';
import url from 'url';

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({});
// History stores the conversation. 
const History = []; 

// Function to rephrase user questions (Forces English for Search)
async function transformQuery(question){
    try {
        const tempHistory = [...History, { role: 'user', parts: [{ text: question }] }];

        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: tempHistory,
            config: {
              systemInstruction: `You are a query rewriting expert. Rephrase the last user question into a standalone question **IN ENGLISH**.
              Even if the user asks in Hindi/Tamil/Hinglish, translate the intent to English for database search.
              If the input is gibberish, random letters (like 'asdfg'), or completely unreadable, output EXACTLY the word: "UNCLEAR".
              Only output the rewritten English question or the word "UNCLEAR".`,
            },
        });
         
        if (!response.text) return "UNCLEAR";
        return response.text;

    } catch (error) {
        console.log("Translation blocked/failed, defaulting to UNCLEAR.");
        return "UNCLEAR"; 
    }
}

// Main Chat Function
async function chatting(question,selectedLang) {
    const queries = await transformQuery(question);
    
    let context = ""; 
    
    const isUnclear = queries.toUpperCase().includes("UNCLEAR");
    
    // 1. Vector Search & Pinecone
    if (!isUnclear) {
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
            console.error("Database search failed:", dbError);
        }
    }

    // 2. Generate Response (BULLETPROOFED)
    
    // Step A: Temporarily add the user's question to history
    History.push({
        role:'user',
        parts:[{text:question}]
    });
    
    try {
        // Step B: Try to get a response from Gemini
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: History,
            config: {
                systemInstruction: `You are 'SmartHealthChat', a warm, empathetic, and thorough virtual health advisor. 
Your goal is to simulate a consultation with a caring, human doctor.

**CRITICAL LANGUAGE OVERRIDE:** The user has selected the language code '${selectedLang}' in their settings. 
You MUST translate and write your final response entirely in the language that matches this code. Ignore the language the user typed in; ALWAYS reply in the language of '${selectedLang}'.

**CRITICAL RULE: DO NOT jump to conclusions. If symptoms are general (like fever, headache, nausea), you MUST ask follow-up questions to narrow down the cause before suggesting a specific disease.**

**Your Guiding Rules (In Order of Importance):**

1.  **Safety First (Immediate Emergencies):**
    * If a user mentions emergency symptoms (chest pain, trouble breathing, severe bleeding, unconsciousness), STOP and advise immediate emergency care in the user's language.

2.  **Strict Domain Restriction (NO General Trivia):**
    * **You are a HEALTH chatbot only.**
    * You do **NOT** know about celebrities, sports, politics, movies, or general news.
    * If a user asks "Who is Virat Kohli?", you MUST refuse.

3.  **Handling Gibberish (The Unclear Input Rule):**
    * If the user types random letters or the input makes absolutely no sense, do NOT try to diagnose them.
    * **Response:** Politely say, "I'm sorry, I didn't quite catch that. Could you please describe your health concern or symptoms clearly?"

4.  **The "Differential Diagnosis" (The Detective Phase):**
    * If the user lists common symptoms (e.g., "I have a fever"), do NOT say "It might be Dengue" immediately.
    * Instead, ask specific questions to distinguish between them.

5.  **Warmth and Empathy:**
    * Always validate their feelings.

6.  **Offer Solutions & Home Remedies:**
    * **ALWAYS** provide safe, home-care advice found in your knowledge base *before* advising a doctor visit.

7.  **Disclaimer:**
    * ALWAYS end with: "Just a quick reminder: I'm an AI assistant and not a medical professional. For personal medical advice, please consult a doctor."

8.  **Context Use:** Use the provided context as your internal knowledge, but do not cite it.

Here is your knowledge base:
Context: ${context}
          `,
            },
        });

        // Step C: If successful, save the AI's reply to history
        History.push({
            role:'model',
            parts:[{text:response.text}]
        });

        return response.text;

    } catch (finalError) {
        // Step D: THE ROLLBACK - If Gemini crashes on the gibberish, catch the error!
        console.error("Gemini crashed on final generation:", finalError);
        
        // Remove the poisoned gibberish from the chat history so it doesn't break future turns
        History.pop(); 
        
        // Send a polite fallback message natively instead of throwing a 500 error to Streamlit
        return "I'm sorry, I didn't quite catch that. Could you please describe your health concern or symptoms clearly?";
    }
}


// --- Server Setup ---
const server = http.createServer(async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json; charset=utf-8');

    const parsedUrl = url.parse(req.url, true);

    if (parsedUrl.pathname === '/ask' && parsedUrl.query.q) {
        const question = parsedUrl.query.q;
        const selectedLang = parsedUrl.query.lang || 'en-IN'; 
        console.log(`Received question: ${question} | Language: ${selectedLang}`);
        
        try {
            const answer = await chatting(question, selectedLang); 
            res.statusCode = 200;
            res.end(JSON.stringify({ answer: answer }));
        } catch (error) {
            console.error("Error processing chat:", error);
            res.statusCode = 500;
            res.end(JSON.stringify({ error: "Sorry, something went wrong." }));
        }
    } else {
        res.statusCode = 404;
        res.end(JSON.stringify({ error: "Not found. Try /ask?q=..." }));
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`SmartHealthChat server running at http://localhost:${PORT}`);
});