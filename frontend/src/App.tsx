import { useState, useCallback } from "react";
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
            const response = await fetch(`${apiUrl}/predict`, {
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


    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
            <div className="container mx-auto px-4 py-6 sm:py-8 lg:py-12">
                <div className="max-w-4xl mx-auto">
                    {/* Header */}
                    <header className="text-center mb-8 sm:mb-12">
                        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 mb-4 sm:mb-6">
                            Real-time Sentiment Analysis
                        </h1>
                        <p className="text-gray-400 text-sm sm:text-base max-w-2xl mx-auto">
                            Analyze the emotional tone of your text in real-time using advanced AI
                        </p>
                    </header>

                    {/* Input Section */}
                    <div className="bg-gray-800/50 backdrop-blur-sm p-4 sm:p-6 lg:p-8 rounded-xl border border-gray-700/50 shadow-2xl mb-6 sm:mb-8">
                        <label htmlFor="text-input" className="block text-sm font-medium text-gray-300 mb-3">
                            Enter your text below:
                        </label>
                        <textarea
                            id="text-input"
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            onKeyUp={() => analyzeSentiment(text)}
                            placeholder="Type your text here to analyze its sentiment..."
                            className="w-full p-4 bg-gray-900/70 border border-gray-600 rounded-lg shadow-inner focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 resize-none text-gray-100 placeholder-gray-500 text-sm sm:text-base"
                            rows={6}
                        />
                        <div className="mt-3 text-xs text-gray-500">
                            {text.length} characters
                        </div>
                    </div>

                    {/* Loading State */}
                    {loading && (
                        <div className="text-center mb-6 sm:mb-8">
                            <div className="inline-flex items-center px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-full">
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400 mr-3"></div>
                                <span className="text-blue-400 text-sm">Analyzing sentiment...</span>
                            </div>
                        </div>
                    )}

                    {/* Results Section */}
                    {result && (
                        <div className="bg-gray-800/50 backdrop-blur-sm p-4 sm:p-6 lg:p-8 rounded-xl border border-gray-700/50 shadow-2xl">
                            <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-center">
                                <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-400 to-blue-500">
                                    Analysis Results
                                </span>
                            </h2>

                            <div className="grid gap-4 sm:gap-6">
                                {/* Sentiment Label */}
                                <div className="bg-gray-900/60 p-4 sm:p-5 rounded-lg border border-gray-700/50">
                                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                                        <span className="font-medium text-gray-300 text-sm sm:text-base">Sentiment:</span>
                                        <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${result.label.toLowerCase() === 'positive'
                                                ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                                                : result.label.toLowerCase() === 'negative'
                                                    ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                                                    : 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                                            }`}>
                                            {result.label}
                                        </span>
                                    </div>
                                </div>

                                {/* Confidence Score */}
                                <div className="bg-gray-900/60 p-4 sm:p-5 rounded-lg border border-gray-700/50">
                                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-3">
                                        <span className="font-medium text-gray-300 text-sm sm:text-base">Confidence Score:</span>
                                        <span className="text-blue-400 font-mono text-sm sm:text-base">
                                            {(result.score * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                        <div
                                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full transition-all duration-500"
                                            style={{ width: `${result.score * 100}%` }}
                                        ></div>
                                    </div>
                                </div>

                                {/* Processing Time */}
                                <div className="bg-gray-900/60 p-4 sm:p-5 rounded-lg border border-gray-700/50">
                                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                                        <span className="font-medium text-gray-300 text-sm sm:text-base">Processing Time:</span>
                                        <span className="text-purple-400 font-mono text-sm sm:text-base">
                                            {result.time_taken} ms
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Empty State */}
                    {!result && !loading && text.trim() === "" && (
                        <div className="text-center py-12 sm:py-16">
                            <div className="text-gray-500 mb-4">
                                <svg className="mx-auto h-12 w-12 sm:h-16 sm:w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 8h10m0 0V6a2 2 0 00-2-2H9a2 2 0 00-2 2v2m10 0v10a2 2 0 01-2 2H9a2 2 0 01-2-2V8m10 0H7" />
                                </svg>
                            </div>
                            <p className="text-gray-400 text-sm sm:text-base">
                                Enter some text above to get started with sentiment analysis
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

}


export default App;


