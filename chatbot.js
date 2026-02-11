const functions = require("firebase-functions");
require('dotenv').config();
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Initialize the Gemini API client
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

exports.chatbotQuery = functions.https.onCall(async (data, context) => {
  // Ensure the user is authenticated
  if (!context.auth) {
    throw new functions.https.HttpsError(
      "unauthenticated",
      "The function must be called while authenticated."
    );
  }

  const userQuery = data.query;
  if (!userQuery) {
    throw new functions.https.HttpsError(
      "invalid-argument",
      "The function must be called with a `query` argument."
    );
  }

  // In the next steps, we will add the RAG logic here.
  // For now, we will just send the query to the Gemini API.

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-pro" });
    const result = await model.generateContent(userQuery);
    const response = await result.response;
    const text = response.text();

    return { response: text };
  } catch (error) {
    console.error("Error generating content with Gemini API:", error);
    throw new functions.https.HttpsError(
      "internal",
      "An error occurred while processing your request."
    );
  }
});