import { useState, useEffect, useCallback } from "react";
import './App.css';


interface SentimentData {
    label: string;
    score: number;
    time_taken: number;
}

function App() {
    const [text, setText] = useState("");
    const [result, setResult] = useState<SentimentData | null>(null);
    const [loading, setLoading] = useState(false);
    const apiUrl = "http://localhost:8000";

    const analyzeSentiment = useCallback(async (inputText: string) => {
        if (!inputText.trim()) {
            setResult(null);
            return;
        }
        setLoading(true);
        try {
            const response = await fetch(`${apiUrl}/sentiment`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: inputText }),
            });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data: SentimentData = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Error fetching sentiment data:', error);
            setResult(null);
        } finally {
            setLoading(false);
        }
    }, [apiUrl]);


    // useEffect(() => {
    //     const handler = setTimeout(() => {
    //         analyzeSentiment(text);
    //     }, 500);
    //     return () => clearTimeout(handler);
    // }, [text, analyzeSentiment]);


    return (
        <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white p-8">
            <div className="max-w-3xl mx-auto">
                <header className="text-center mb-12">
                    <h1 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 mb-8">
                        Real-time Sentiment Analysis
                    </h1>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        onKeyUp={() => analyzeSentiment(text)}
                        placeholder="Type your text here..."
                        className="w-full p-4 bg-gray-800 border border-gray-700 rounded-lg shadow-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-200 placeholder-gray-500"
                        rows={5}
                    />
                    {loading && (
                        <div className="mt-6 text-blue-400">
                            <p className="animate-pulse">Analyzing...</p>
                        </div>
                    )}
                    {result && (
                        <div className="mt-8 bg-gray-800 p-6 rounded-lg border border-gray-700 shadow-xl">
                            <h2 className="text-2xl font-semibold mb-4 text-blue-400">Sentiment Result</h2>
                            <div className="space-y-3">
                                <p className="flex items-center justify-between bg-gray-900 p-3 rounded">
                                    <span className="font-medium">Label:</span>
                                    <span className="text-blue-400">{result.label}</span>
                                </p>
                                <p className="flex items-center justify-between bg-gray-900 p-3 rounded">
                                    <span className="font-medium">Score:</span>
                                    <span className="text-blue-400">{result.score}</span>
                                </p>
                                <p className="flex items-center justify-between bg-gray-900 p-3 rounded">
                                    <span className="font-medium">Time Taken:</span>
                                    <span className="text-blue-400">{result.time_taken} ms</span>
                                </p>
                            </div>
                        </div>
                    )}
                </header>
            </div>
        </div>
    );

}


export default App;


