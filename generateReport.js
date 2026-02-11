const functions = require("firebase-functions");
const { onCall } = require("firebase-functions/v2/https");
const { GoogleGenerativeAI } = require("@google/generative-ai");

// Initialize the Gemini model
exports.generateReport = onCall({ enforceAppCheck: false }, async (request) => {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_KEY);
  const model = genAI.getGenerativeModel({ model: "gemini-pro" });
  const { chartType, data } = request.data;

  let prompt;

  // Create a specific prompt based on the chart type
  switch (chartType) {
    case "userDemographics":
      prompt = `Analyze the following user demographics data and provide a brief, insightful summary in one paragraph. The data is: ${JSON.stringify(data)}. Focus on the distribution and any notable points.`;
      break;
    case "incidentsByTime":
      prompt = `Analyze the following incident report data, which shows incidents over time, and provide a brief, insightful summary in one paragraph. The data is: ${JSON.stringify(data)}. Highlight any trends, peaks, or unusual activity.`;
      break;
    case "campOccupancy":
      prompt = `Analyze the following camp logistics data, which shows camp occupancy vs. capacity, and provide a brief, insightful summary in one paragraph. The data is: ${JSON.stringify(data)}. Comment on the overall capacity and any camps that are nearing their limit.`;
      break;
    case "systemPerformance":
      prompt = `Analyze the following system performance data, which includes API response times and error rates, and provide a brief, insightful summary in one paragraph. The data is: ${JSON.stringify(data)}. Comment on the system's stability and any potential performance issues.`;
      break;
    default:
      return { summary: "Invalid chart type." };
  }

  try {
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const summary = response.text();
    return { summary };
  } catch (error) {
    console.error("Error generating report:", error.message);
    if (error.response) {
      console.error("Error response:", error.response.data);
    }
    return { summary: "Could not generate a summary at this time." };
  }
});