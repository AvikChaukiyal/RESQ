const {onRequest} = require("firebase-functions/v2/https");
const { GoogleGenerativeAI } = require('@google/generative-ai');
const { Pinecone } = require('@pinecone-database/pinecone');
const cors = require('cors')({origin: true});

exports.chatWithRag = onRequest({ secrets: ["GEMINI_KEY", "PINECONE_KEY"] }, (req, res) => {
  cors(req, res, async () => {
  console.log('Executing rag_chatbot.js version: 2025-12-07T18:51:36Z');
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_KEY,
  });
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_KEY);
  if (req.method !== 'POST') {
    return res.status(405).send('Method Not Allowed');
  }

  const { message } = req.body;

  if (!message) {
    return res.status(400).send('Missing message in request body');
  }

  try {
    // 1. Get embeddings for the user's message
    const model = genAI.getGenerativeModel({ model: "text-embedding-004"});
    const embedding = await model.embedContent(message);

    // 2. Query Pinecone for relevant documents
    const index = pinecone.index('resq-docs');
    const queryRequest = {
      vector: embedding.embedding.values,
      topK: 3,
      includeValues: true,
    };
    const queryResponse = await index.query(queryRequest);

    // 3. Augment the prompt with the retrieved documents
    const context = queryResponse.matches.map(match => match.id).join('\n');
    const augmentedPrompt = `
      Context:
      ${context}

      User's question:
      ${message}
    `;

    // 4. Send the augmented prompt to Gemini
    const chatModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest"});
    const result = await chatModel.generateContent(augmentedPrompt);
    const response = await result.response;
    const text = response.text();

    console.log('Gemini response:', text);
    return res.status(200).json({ response: text });
  } catch (error) {
    console.error('Error calling RAG chatbot function:', error);
    return res.status(500).send('Internal Server Error');
  }
  });
});