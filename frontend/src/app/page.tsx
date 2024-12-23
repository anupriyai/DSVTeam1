"use client";

import React, { useState, useEffect } from "react";
import Papa from "papaparse";

const Page = () => {
  const [categories, setCategories] = useState<string[]>([]);
  const [prompts, setPrompts] = useState<Record<string, string[]>>({});
  const [responses, setResponses] = useState<
    Record<string, Record<string, Record<string, string>>>
  >({});
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedPrompt, setSelectedPrompt] = useState("");
  const [selectedTab, setSelectedTab] = useState("database"); // Tabs state
  const [showResponses, setShowResponses] = useState(false); // Control when to show responses
  const [userPrompt, setUserPrompt] = useState(""); // Custom prompt input
  const [userCategories, setUserCategories] = useState<string[]>([]); // Selected categories for user prompt
  const [userResponses, setUserResponses] = useState<Record<string, string>>({
    "GPT-4o": "",
    "Gemini": "",
    "Claude 3.5 Sonnet": "",
    "Llama": "",
  }); // Custom LLM responses
  const [errors, setErrors] = useState<string[]>([]); // Validation errors
  const [showUserResponses, setShowUserResponses] = useState(false); // Show custom responses after validation
  const [serverResponse, setServerResponse] = useState<Record<string, Record<string, Number>>>({});
  const [loading, setLoading] = useState(false);

  const csvUrl =
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTA28iC6m2xcAaGa2RCEIRQs2Oe9hjC738EQQi8rWi7y_iW7mme-HZOQNJIXv_YEQ/pub?output=csv";

  useEffect(() => {
    // Fetch CSV and parse it
    fetch(csvUrl)
      .then((response) => response.text())
      .then((csvText) => {
        const data = Papa.parse(csvText, { header: true });
        const jsonData = data.data;

        const catMap: Record<string, string[]> = {};
        const respMap: Record<string, Record<string, Record<string, string>>> = {};

        jsonData.forEach((row: any) => {
          const category = row.Category;
          const prompt = row.Prompt;

          if (!catMap[category]) {
            catMap[category] = [];
            respMap[category] = {};
          }
          catMap[category].push(prompt);

          respMap[category][prompt] = {
            "GPT-4o": row["GPT-4o"],
            "Gemini": row["Gemini"],
            "Claude 3.5 Sonnet": row["Claude 3.5 Sonnet"],
            "Llama": row["Llama"],
          };
        });

        setCategories(Object.keys(catMap));
        setPrompts(catMap);
        setResponses(respMap);
        setSelectedCategory(Object.keys(catMap)[0]);
        setSelectedPrompt(catMap[Object.keys(catMap)[0]][0]);
      });
  }, []);

  const [message, setMessage] = useState("Loading");

  useEffect(() => {
    fetch("http://localhost:8080/api/score")
      .then((response) => response.json())
      .then((data) => {setMessage(data.message);
      });
  }, []);

  const handleValidation = () => {
    const validationErrors = [];

    if (!userPrompt.trim()) {
      validationErrors.push("You must enter a prompt.");
    }
    if (userCategories.length === 0) {
      validationErrors.push("You must select at least one category.");
    }
    for (const model in userResponses) {
      if (!userResponses[model].trim()) {
        validationErrors.push(`You must enter a response for ${model}.`);
      }
    }

    setErrors(validationErrors);

    return validationErrors.length === 0; // Return true if no errors
  };
  

  const handleCompare = () => {
    if (handleValidation()) {
      sendCustomToBackend();
      setShowUserResponses(true);
    } else {
      setShowUserResponses(false);
    }
  };
  // Send the data of personal prompt to the backend
  const sendCustomToBackend = async () => {
    setLoading(true); 
    const response = await fetch('http://localhost:8080/api/accuracy', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: userPrompt,
        categories: userCategories,
        responses: userResponses,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      console.error('Error:', error);
      setLoading(false);
      return;
    }
    // Backend sends back the custom score
    const result = await response.json();
    console.log(result);
    setServerResponse(result.message);
    setLoading(false);
  };

  // Send to backend after generate button clicked
  const sendPresetToBackend = async () => {
    setLoading(true);
    const response = await fetch('http://localhost:8080/api/accuracy', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: selectedPrompt,
        categories: [selectedCategory],
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      console.error('Error:', error);
      setLoading(false);
      return;
    }
    // Backend sends back the custom score
    const result = await response.json();
    console.log(result);
    setServerResponse(result.message);
    setLoading(false);
    setShowResponses(true);
  }

  const handleTabChange = (tab: string) => {
    setSelectedTab(tab);
    setServerResponse({}); // Clear server response
    setShowResponses(false); // Reset responses visibility
    setShowUserResponses(false); // Reset custom responses visibility
    setLoading(false); // Reset loading state
  };
  

  return (
    <main className="min-h-screen bg-gray-900 text-white p-6">
      <h1 className="text-2xl font-bold mb-4">LLM Evaluation Platform</h1>
      <p className="mb-4">Choose how you would like to test our prompts:</p>

      {/* Tabs */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => handleTabChange("database")}
          className={`p-2 rounded-md ${
            selectedTab === "database" ? "bg-blue-600" : "bg-gray-800"
          }`}
        >
          Choose from Our Database
        </button>
        <button
          onClick={() => handleTabChange("custom")}
          className={`p-2 rounded-md ${
            selectedTab === "custom" ? "bg-blue-600" : "bg-gray-800"
          }`}
        >
          Enter Your Own Prompt
        </button>
      </div>

      {selectedTab === "database" && (
        <>
          {/* Database Tab */}
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            {/* Category Dropdown */}
            <div className="flex flex-col">
              <label htmlFor="category" className="mb-2 font-medium">
                Select Category:
              </label>
              <select
                id="category"
                value={selectedCategory}
                onChange={(e) => {
                  const newCategory = e.target.value;
                  setSelectedCategory(newCategory);
                  setSelectedPrompt(prompts[newCategory][0]);
                  setShowResponses(false); // Reset responses visibility
                }}
                className="bg-gray-800 p-2 rounded-md border border-gray-700"
              >
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>

            {/* Prompt Dropdown */}
            <div className="flex flex-col">
              <label htmlFor="prompt" className="mb-2 font-medium">
                Select Prompt:
              </label>
              <select
                id="prompt"
                value={selectedPrompt}
                onChange={(e) => {
                  setSelectedPrompt(e.target.value);
                  setShowResponses(false); // Reset responses visibility
                }}
                className="bg-gray-800 p-2 rounded-md border border-gray-700"
              >
                {prompts[selectedCategory]?.map((prompt) => (
                  <option key={prompt} value={prompt}>
                    {prompt}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Generate Button */}
          <button
            onClick={() => sendPresetToBackend()}
            className="mb-4 bg-blue-600 px-4 py-2 rounded-md"
          >
            Generate
          </button>

          {/* LLM Output Comparison */}
          {loading ? (
            <div>Loading...</div>
          ) : (
          showResponses && (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              {["GPT-4o", "Gemini", "Claude 3.5 Sonnet", "Llama"].map((model) => (
                <div
                  key={model}
                  className="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-md"
                >
                  <h2 className="text-lg font-semibold mb-2">{model}</h2>
                  <p className="text-sm">
                    Cluster Accuracy: {serverResponse[model]?.cluster_acc.toString() || "No data available"}
                  </p>
                  <p className="text-sm">
                    Accuracy: {serverResponse[model]?.accuracy.toString()  || "No data available"}
                  </p>
                  <p className="text-sm">
                    Coherence: {serverResponse[model]?.coherence.toString() || "No data available"}
                  </p>
                  <p className="text-sm">
                    Relevance: {serverResponse[model]?.relevance.toString() || "No data available"}
                  </p>
                  <p className="text-sm">
                    Creativity: {serverResponse[model]?.creativity.toString() || "No data available"}
                  </p>
                  <p className="text-sm">
                    Bias: {serverResponse[model]?.bias.toString() || "No data available"}
                  </p>
                  <p className="text-sm">
                    Custom: {serverResponse[model]?.custom.toString() || "No data available"}
                  </p>
                </div>
              ))}
            </div>
          )
          )}
        </>
      )}

      {selectedTab === "custom" && (
        <>
          {/* Custom Prompt Tab */}
          <div className="flex flex-col mb-6">
            <label htmlFor="customPrompt" className="mb-2 font-medium">
              Enter Your Prompt:
            </label>
            <input
              id="customPrompt"
              type="text"
              value={userPrompt}
              onChange={(e) => setUserPrompt(e.target.value)}
              className="bg-gray-800 p-2 rounded-md border border-gray-700 mb-4"
              placeholder="Type your prompt here..."
            />

            {/* Category Selector */}
            <label className="mb-2 font-medium">Classify Your Prompt:</label>
            <div className="flex flex-wrap gap-2 mb-4">
              {categories.map((category) => (
                <div key={category} className="flex items-center">
                  <input
                    type="radio"
                    value={category}
                    name="category"  // Same name ensures only one can be selected
                    onChange={(e) => setUserCategories([e.target.value])}  // Set only the selected category
                    checked={userCategories.includes(category)}  // Check if the category is selected
                    className="mr-2"
                  />
                  <label>{category}</label>
                </div>
              ))}
            </div>

            {/* LLM Response Inputs */}
            <label className="mb-2 font-medium">LLM Responses:</label>
            {["GPT-4o", "Gemini", "Claude 3.5 Sonnet", "Llama"].map((model) => (
              <div key={model} className="mb-4">
                <label htmlFor={model} className="block mb-1">
                  {model} Response:
                </label>
                <textarea
                  id={model}
                  value={userResponses[model]}
                  onChange={(e) =>
                    setUserResponses({
                      ...userResponses,
                      [model]: e.target.value,
                    })
                  }
                  className="bg-gray-800 p-2 rounded-md border border-gray-700 w-full"
                  rows={3}
                  placeholder={`Enter ${model} response here...`}
                ></textarea>
              </div>
            ))}
          </div>

          {/* Error Messages */}
          {errors.length > 0 && (
            <div className="mb-4 text-red-500">
              {errors.map((error, index) => (
                <p key={index}>{error}</p>
              ))}
            </div>
          )}

          {/* Compare Button */}
          <button
            onClick={handleCompare}
            className="mb-4 bg-blue-600 px-4 py-2 rounded-md"
          >
            Compare
          </button>

          {/* Display Custom LLM Output Comparison */}
          {loading ? (
            <div>Loading...</div>
          ) : (
            showUserResponses && (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
                {["GPT-4o", "Gemini", "Claude 3.5 Sonnet", "Llama"].map((model) => (
                  <div
                    key={model}
                    className="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-md"
                  >
                    <h2 className="text-lg font-semibold mb-2">{model}</h2>
                    <p className="text-sm">
                    Cluster Accuracy: {serverResponse[model]?.cluster_acc.toString() || "No data available"}
                    </p>
                    <p className="text-sm">
                      Accuracy: {serverResponse[model]?.accuracy.toString()  || "No data available"}
                    </p>
                    <p className="text-sm">
                      Coherence: {serverResponse[model]?.coherence.toString() || "No data available"}
                    </p>
                    <p className="text-sm">
                      Relevance: {serverResponse[model]?.relevance.toString() || "No data available"}
                    </p>
                    <p className="text-sm">
                      Creativity: {serverResponse[model]?.creativity.toString() || "No data available"}
                    </p>
                    <p className="text-sm">
                      Bias: {serverResponse[model]?.bias.toString() || "No data available"}
                    </p>
                    <p className="text-sm">
                      Custom: {serverResponse[model]?.custom.toString() || "No data available"}
                    </p>
                  </div>
                ))}
              </div>
            )
          )}
        </>
      )}
    </main>
  );
};

export default Page;
