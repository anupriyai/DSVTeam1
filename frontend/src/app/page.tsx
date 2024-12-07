"use client";

import React, { useState, useEffect } from "react";
import Papa from "papaparse";

interface CsvRow {
  Category: string;
  Prompt: string;
  "GPT-4o": string;
  "Gemini": string;
  "Claude 3.5 Sonnet": string;
  "Llama": string;
}

const Page = () => {
  const [categories, setCategories] = useState<string[]>([]);
  const [prompts, setPrompts] = useState<Record<string, string[]>>({});
  const [responses, setResponses] = useState<
    Record<string, Record<string, Record<string, string>>>
  >({});
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedPrompt, setSelectedPrompt] = useState("");

  const csvUrl =
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vTA28iC6m2xcAaGa2RCEIRQs2Oe9hjC738EQQi8rWi7y_iW7mme-HZOQNJIXv_YEQ/pub?output=csv";

  useEffect(() => {
    // Fetch CSV and parse it
    fetch(csvUrl)
      .then((response) => response.text())
      .then((csvText) => {
        const data = Papa.parse(csvText, { header: true });
        const jsonData = data.data as CsvRow[]; // Cast to CsvRow[]

        // Process data
        const catMap: Record<string, string[]> = {};
        const respMap: Record<string, Record<string, Record<string, string>>> = {};

        jsonData.forEach((row: CsvRow) => {
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

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <h1 className="text-2xl font-bold mb-4">LLM Evaluation Interface</h1>

      {categories.length > 0 && (
        <>
          {/* Category and Prompt Selector */}
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
                onChange={(e) => setSelectedPrompt(e.target.value)}
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

          {/* LLM Output Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            {["GPT-4o", "Gemini", "Claude 3.5 Sonnet", "Llama"].map((model) => (
              <div
                key={model}
                className="bg-gray-800 p-4 rounded-lg border border-gray-700 shadow-md"
              >
                <h2 className="text-lg font-semibold mb-2">{model}</h2>
                <p className="text-sm">
                  {responses[selectedCategory]?.[selectedPrompt]?.[model] || "No data available"}
                </p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};

export default Page;
