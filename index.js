//PDF LOADING
import * as dotenv from 'dotenv';
dotenv.config();

// --- NEW IMPORTS ---
// We need 'fs' (File System) to read folders
// and 'path' to correctly join file paths.
import * as fs from 'fs';
import * as path from 'path';
// --- END NEW IMPORTS ---

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

async function indexDocument() {
    // --- THIS IS THE UPDATED SECTION ---

    // 1. Point to your new folder
    const docsPath = './who';

    // 2. Read all filenames in the folder that end with ".pdf"
    const fileNames = fs.readdirSync(docsPath).filter(file => file.endsWith('.pdf'));
    
    // 3. Create an empty array to hold the content from all PDFs
    let rawDocs = [];

    console.log(`Starting to load ${fileNames.length} PDF files from '${docsPath}'...`);

    // 4. Loop through each file
    for (const file of fileNames) {
        const filePath = path.join(docsPath, file);
        const pdfLoader = new PDFLoader(filePath);
        
        try {
            const docs = await pdfLoader.load();
            rawDocs = rawDocs.concat(docs); // Add the docs from this file to the main array
            console.log(`Successfully loaded ${file}`);
        } catch (error) {
            // This will catch any corrupted or problematic PDFs
            console.error(`Failed to load ${file}. Skipping. Error: ${error.message}`);
        }
    }

    console.log("All PDF documents have been loaded.");
    // --- END OF UPDATED SECTION ---


    // --- The rest of your code stays exactly the same ---

    console.log("Chunking documents...");
    //chunking document
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log("Chunking done.");

    // converting to vector 
    console.log("Initializing embedding model...");
    const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'gemini-embedding-001', // We must use this exact string format for the LangChain wrapper
    taskType: "RETRIEVAL_DOCUMENT" // This tells Google we are storing documents
});
    console.log("Embedding model ready.");

    //database ko bhi configure
    //initialise pinecone client
    console.log("Connecting to Pinecone...");
    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
    console.log("Pinecone config done.");

    //langchain(chunking,embedding model,database)
    console.log("Storing vectors in Pinecone. This may take a while...");
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
    });
    console.log("Data stored successfully in Pinecone!");
}

indexDocument();