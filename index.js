import * as dotenv from 'dotenv';
dotenv.config();

import * as fs from 'fs';
import * as path from 'path';

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Pinecone } from '@pinecone-database/pinecone';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ✅ Uses HuggingFace FREE inference API — no daily limit, no billing
// Model: all-MiniLM-L6-v2 → produces 384-dim vectors
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

    if (!response.ok) {
        const err = await response.text();
        throw new Error(`HF Error ${response.status}: ${err.slice(0, 100)}`);
    }

    const data = await response.json();
    // HuggingFace returns a nested array [[...]] for single input
    return Array.isArray(data[0]) ? data[0] : data;
}

async function indexDocument() {
    console.log("=========================================");
    console.log("🚀 INDEXING WITH HUGGINGFACE (NO LIMITS)");
    console.log("=========================================\n");

    if (!process.env.HF_API_KEY) {
        console.error("❌ HF_API_KEY missing from .env!");
        console.error("Get a FREE key at: https://huggingface.co/settings/tokens");
        process.exit(1);
    }

    const docsPath = path.resolve('./who');
    const fileNames = fs.readdirSync(docsPath).filter(f => f.endsWith('.pdf'));
    console.log(`[STEP 1] Loading ${fileNames.length} PDFs...`);

    let rawDocs = [];
    for (const file of fileNames) {
        try {
            const docs = await new PDFLoader(path.join(docsPath, file)).load();
            rawDocs = rawDocs.concat(docs);
        } catch (e) {
            console.warn(`  ⚠️ Skipping ${file}`);
        }
    }

    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
    let chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    chunkedDocs = chunkedDocs.filter(doc => doc.pageContent.replace(/\s/g, '').length > 50);
    console.log(`✅ Total chunks: ${chunkedDocs.length}\n`);

    // Test embedding
    console.log("[STEP 2] Testing HuggingFace embedding...");
    try {
        const testVec = await getEmbedding("test health query");
        console.log(`✅ Works! Dimension = ${testVec.length}`);
        console.log(`⚠️  IMPORTANT: Your Pinecone index must be dimension ${testVec.length}`);
        console.log(`   If your index is 768-dim, you need to recreate it as ${testVec.length}-dim\n`);
    } catch (e) {
        console.error("❌ HuggingFace test failed:", e.message);
        process.exit(1);
    }

    const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    console.log("=========================================");
    console.log(`⏳ UPLOADING ${chunkedDocs.length} CHUNKS...`);
    console.log("=========================================\n");

    const BATCH_SIZE = 50;
    let totalUploaded = 0;
    const totalBatches = Math.ceil(chunkedDocs.length / BATCH_SIZE);

    for (let i = 0; i < chunkedDocs.length; i += BATCH_SIZE) {
        const batch = chunkedDocs.slice(i, i + BATCH_SIZE);
        const batchNum = Math.floor(i / BATCH_SIZE) + 1;
        process.stdout.write(`📦 Batch ${batchNum}/${totalBatches}... `);

        const vectorsToUpload = [];

        for (let j = 0; j < batch.length; j++) {
            let retries = 0;
            while (retries < 3) {
                try {
                    const vector = await getEmbedding(batch[j].pageContent);
                    vectorsToUpload.push({
                        id: `chunk-${i + j}`,
                        values: vector,
                        metadata: {
                            text: batch[j].pageContent,
                            source: batch[j].metadata?.source || 'unknown',
                        }
                    });
                    break;
                } catch (err) {
                    retries++;
                    await sleep(2000 * retries);
                }
            }
            await sleep(100);
        }

        if (vectorsToUpload.length > 0) {
            await pineconeIndex.upsert(vectorsToUpload);
            totalUploaded += vectorsToUpload.length;
        }

        const pct = ((i + batch.length) / chunkedDocs.length * 100).toFixed(1);
        console.log(`✅ ${totalUploaded} uploaded (${pct}%)`);
        await sleep(500);
    }

    console.log("\n=========================================");
    console.log(`🎉 DONE! Uploaded ${totalUploaded} chunks`);
    console.log("=========================================");
}

indexDocument().catch(err => {
    console.error("\n💥 FATAL ERROR:", err.message);
    process.exit(1);
});