import React from 'react';
import { motion } from 'framer-motion';
import { ScanText, Database, ServerOff, Search, Cpu, FileJson } from 'lucide-react';

interface FeatureData {
  icon: React.ReactNode;
  title: string;
  description: string;
  color: string;
}

const featureList: FeatureData[] = [
  {
    icon: <ServerOff className="w-6 h-6" />,
    title: "Privacy-First RAG",
    description: "Grounds AI answers in your specific files via Ollama. Optionally supports secure APIs (like Gemini) if user-configured.",
    color: "group-hover:text-neon"
  },
  {
    icon: <ScanText className="w-6 h-6" />,
    title: "High-Fidelity OCR",
    description: "Digitize scanned PDFs and images using Tesseract/EasyOCR, automatically extracting metadata like invoices and dates.",
    color: "group-hover:text-alert"
  },
  {
    icon: <Search className="w-6 h-6" />,
    title: "Hybrid Vector Search",
    description: "Combines FAISS for fast semantic similarity with SQL-based storage for structured history and precise retrieval.",
    color: "group-hover:text-blue-400"
  },
  {
    icon: <FileJson className="w-6 h-6" />,
    title: "Smart Extraction",
    description: "Proactively flags upcoming deadlines and exports structured summaries to CSV or JSON formats.",
    color: "group-hover:text-purple-400"
  },
  {
    icon: <Cpu className="w-6 h-6" />,
    title: "Local Optimization",
    description: "Integration of lighter quantized models ensures smooth performance even on standard consumer laptops.",
    color: "group-hover:text-emerald-400"
  },
  {
    icon: <Database className="w-6 h-6" />,
    title: "Multi-Modal Future",
    description: "Roadmap support for analyzing diagrams, charts, and handwritten notes directly within the chat interface.",
    color: "group-hover:text-yellow-400"
  }
];

export const Features = () => {
  return (
    <section id="features" className="py-32 px-6 bg-charcoal relative">
      <div className="max-w-7xl mx-auto">
        <div className="mb-20">
          <h2 className="font-display text-5xl md:text-7xl font-bold uppercase text-white mb-6">
            System <span className="text-neon">Capabilities</span>
          </h2>
          <div className="w-full h-[1px] bg-white/10"></div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-1">
          {featureList.map((feature, index) => (
            <FeatureCard key={index} feature={feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
};

interface FeatureCardProps {
  feature: FeatureData;
  index: number;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ feature, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}
      className="group relative bg-obsidian p-8 md:p-12 border border-white/5 hover:border-white/20 transition-all duration-300 min-h-[300px] flex flex-col justify-between"
    >
      <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity">
        <span className="font-mono text-xs text-white/30">0{index + 1}</span>
      </div>

      <div className={`text-white/50 transition-colors duration-300 ${feature.color} mb-6`}>
        {feature.icon}
      </div>

      <div>
        <h3 className="font-display text-2xl font-bold text-white mb-4 uppercase tracking-wide">
          {feature.title}
        </h3>
        <p className="font-mono text-sm text-steel leading-relaxed">
          {feature.description}
        </p>
      </div>

      <div className="absolute bottom-0 left-0 w-0 h-[2px] bg-neon group-hover:w-full transition-all duration-500"></div>
    </motion.div>
  );
};