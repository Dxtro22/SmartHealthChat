import * as dotenv from 'dotenv';
dotenv.config();

import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';

// 1. Setup AI translation tool
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'gemini-embedding-001', 
});

// 2. Connect to Pinecone
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// 3. The MEGA Medical Dataset (Categorized & Optimized)
const medicalData = [
    // --- CRITICAL EMERGENCIES ---
    "Heart Attack: Severe chest pain radiating to the left arm, neck, or jaw, accompanied by shortness of breath and cold sweat. Call emergency services instantly.",
    "Stroke: Face drooping on one side, sudden arm weakness, and slurred speech (F.A.S.T. signs). Requires immediate emergency transport.",
    "Anaphylaxis: Severe allergic reaction causing swelling of the throat, difficulty breathing, and hives. Use an epinephrine auto-injector (EpiPen) immediately.",
    "Seizure: Sudden, uncontrolled body movements or loss of consciousness. Clear the area of sharp objects, place the person on their side, and do NOT put anything in their mouth.",

    // --- COMMON INFECTIONS & TROPICAL DISEASES ---
    "Dengue Fever: Transmitted by mosquitoes. Symptoms include sudden high fever, severe joint/muscle pain (breakbone fever), pain behind the eyes, and skin rash. Hydration and monitoring platelets are critical.",
    "Malaria: Mosquito-borne disease causing high fever with severe chills, sweating, and headache. Requires immediate blood testing and antimalarial medication.",
    "Typhoid: Bacterial infection from contaminated food/water. Symptoms include prolonged high fever, weakness, stomach pain, and sometimes a rose-colored rash. Treatable with antibiotics.",
    "Urinary Tract Infection (UTI): Causes a strong, persistent urge to urinate, and a burning sensation during urination. Drinking plenty of water and antibiotics are the standard treatment.",

    // --- RESPIRATORY ISSUES ---
    "Asthma Attack: Sudden feeling of chest tightness, severe coughing, and wheezing. The patient should immediately use their prescribed rescue inhaler (Albuterol).",
    "Pneumonia: Infection inflaming the air sacs in one or both lungs. Symptoms include cough with phlegm, fever, chills, and difficulty breathing.",
    "Common Cold: Viral infection causing runny nose, sore throat, sneezing, and mild cough. Best treated with rest, warm liquids, and over-the-counter decongestants.",
    "Influenza (Flu): Comes on suddenly with high fever, severe body aches, chills, and extreme fatigue. Antiviral drugs can help if taken early.",

    // --- GASTROINTESTINAL ISSUES ---
    "Food Poisoning: Caused by eating contaminated food. Symptoms include stomach cramps, diarrhea, and vomiting within 6 to 24 hours. The most important treatment is drinking ORS to prevent dehydration.",
    "Acid Reflux / GERD: A burning pain in the lower chest (heartburn) occurring after eating. Avoid spicy, fatty foods and do not lie down immediately after meals.",
    "Appendicitis: Sudden, sharp pain that begins around the belly button and shifts to the lower right abdomen. Accompanied by nausea and fever. Requires emergency surgery.",

    // --- NEUROLOGICAL & HEADACHES ---
    "Migraine: Intense, throbbing pain usually on one side of the head, accompanied by nausea and extreme sensitivity to light and sound. Rest in a dark, quiet room.",
    "Tension Headache: A dull, aching head pain that feels like a tight band around the head. Caused by stress or poor posture. Treat with hydration and relaxation.",
    "Concussion: A mild traumatic brain injury after a bump to the head. Symptoms include confusion, dizziness, headache, and nausea. Requires physical and mental rest.",

    // --- FIRST AID & INJURIES ---
    "Minor Burns: Run cool (not ice cold) water over the affected skin for 10-15 minutes. Never apply butter, ice, or oil, as it traps the heat.",
    "Sprains (Ankle/Wrist): Follow the R.I.C.E. method: Rest the joint, apply Ice for 20 minutes, use Compression bandages, and Elevate the limb above the heart.",
    "Nosebleeds: Sit upright and lean slightly forward (not backward). Pinch the soft part of the nose continuously for 10 to 15 minutes.",
    "Deep Cuts: Apply firm, direct pressure to the wound using a clean cloth or bandage until the bleeding stops. If it doesn't stop after 10 minutes, seek emergency care.",

    // --- SKIN CONDITIONS ---
    "Eczema: Red, itchy, and inflamed patches of skin. Triggered by allergens or stress. Keep skin highly moisturized and avoid harsh soaps.",
    "Sunburn: Red, painful skin that feels hot to the touch after UV exposure. Apply aloe vera gel, take cool baths, and drink plenty of water.",

    // --- MENTAL HEALTH & WELLNESS ---
    "Panic Attack: Sudden episode of intense fear causing rapid heart rate, shortness of breath, and chest pain. Practice the 4-7-8 deep breathing technique to calm the nervous system.",
    "Dehydration: Causes fatigue, dizziness, dry mouth, and dark-colored urine. Prevent this by drinking at least 8 large glasses of water daily."
];

async function uploadToPinecone() {
    console.log("=========================================");
    console.log("Starting Mega-Upload to Pinecone...");
    console.log(`Total diseases & protocols to learn: ${medicalData.length}`);
    console.log("=========================================\n");
    
    const vectorsToUpload = [];

    try {
        // Loop through each fact, translate it, and prepare for upload
        for (let i = 0; i < medicalData.length; i++) {
            const textFact = medicalData[i];
            console.log(`[${i + 1}/${medicalData.length}] Translating data into vectors...`);
            
            // Send to Gemini to get the 3072 numbers
            const vectorNumbers = await embeddings.embedQuery(textFact); 
            
            vectorsToUpload.push({
                id: `medical-fact-${i}`,
                values: vectorNumbers,
                metadata: { text: textFact } // Attach the English sticky note
            });
        }

        console.log("\nUploading all vectors to Pinecone database...");
        await pineconeIndex.upsert(vectorsToUpload);
        
        console.log("=========================================");
        console.log("✅ SUCCESS! Your AI Doctor is now fully trained.");
        console.log("=========================================");

    } catch (error) {
        console.error("❌ An error occurred during upload:", error);
    }
}

uploadToPinecone();