import * as dotenv from 'dotenv';
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

async function diagnose() {
    console.log("🔍 RUNNING DIAGNOSTICS...\n");

    // Test 1: Check which models work and what dimension they return
    const modelsToTest = [
        'models/text-embedding-004',
        'text-embedding-004',
        'models/embedding-001',
        'embedding-001',
        'gemini-embedding-001',
        'models/gemini-embedding-001',
    ];

    for (const modelName of modelsToTest) {
        try {
            const emb = new GoogleGenerativeAIEmbeddings({
                apiKey: process.env.GEMINI_API_KEY,
                model: modelName,
            });
            const vec = await emb.embedQuery("test health query");
            console.log(`✅ Model "${modelName}" → dimension: ${vec.length}`);
        } catch (e) {
            console.log(`❌ Model "${modelName}" → ERROR: ${e.message.slice(0, 80)}`);
        }
    }

    // Test 2: Check Pinecone index dimension
    console.log("\n🔍 Checking Pinecone index...");
    try {
        const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
        const indexList = await pinecone.listIndexes();
        console.log("Indexes found:", JSON.stringify(indexList, null, 2));
    } catch (e) {
        console.log("❌ Pinecone error:", e.message);
    }
}

diagnose().catch(console.error);
