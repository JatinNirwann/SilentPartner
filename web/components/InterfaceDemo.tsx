import React from 'react';
import { motion } from 'framer-motion';
import { Cpu, FileText, CheckCircle2, Terminal, Shield } from 'lucide-react';

export const InterfaceDemo = () => {
  return (
    <section id="interface" className="py-32 px-6 bg-obsidian border-t border-white/10 overflow-hidden relative">
       {/* Ambient Light */}
       <div className="absolute top-1/2 right-0 -translate-y-1/2 w-[500px] h-[500px] bg-blue-900/10 blur-[120px] rounded-full pointer-events-none"></div>

      <div className="max-w-7xl mx-auto flex flex-col lg:flex-row items-center gap-20">
        <div className="lg:w-1/2">
          <h2 className="font-display text-5xl md:text-7xl font-bold uppercase text-white mb-8">
            Talk to Your<br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-500">Data Directly</span>
          </h2>
          <p className="font-mono text-steel text-lg mb-8 leading-relaxed">
            SilentPartner transforms your unorganized document folder into a structured, searchable knowledge base. 
          </p>
          <p className="font-mono text-steel text-lg mb-12 leading-relaxed">
            Ask complex questions about your legal contracts, financial statements, or medical records. The answers are generated locally, ensuring your secrets remain yours.
          </p>

          <ul className="space-y-4 font-mono text-sm text-white">
             <li className="flex items-center gap-4">
                <span className="text-neon"><Cpu size={20} /></span>
                <span>Running Llama-3-8b-Quantized (Local)</span>
             </li>
             <li className="flex items-center gap-4">
                <span className="text-neon"><Shield size={20} /></span>
                <span>Air-gapped Environment Supported</span>
             </li>
             <li className="flex items-center gap-4">
                <span className="text-neon"><FileText size={20} /></span>
                <span>PDF, DOCX, & Image OCR Ingestion</span>
             </li>
          </ul>
        </div>

        <div className="lg:w-1/2 relative flex justify-center">
          {/* Device Frame */}
          <div className="relative w-[350px] h-[700px] bg-black border-[1px] border-white/20 rounded-[3rem] p-4 shadow-2xl shadow-neon/5">
            {/* Screen */}
            <div className="w-full h-full bg-charcoal rounded-[2.5rem] overflow-hidden relative flex flex-col font-mono">
              
              {/* App Header */}
              <div className="h-16 border-b border-white/10 flex items-center justify-between px-6 bg-obsidian z-10">
                 <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-neon rounded-full animate-pulse"></div>
                    <span className="text-xs text-neon uppercase tracking-widest">Online (Local)</span>
                 </div>
                 <Terminal size={16} className="text-white/40" />
              </div>

              {/* Chat Area */}
              <div className="flex-1 p-6 space-y-6 overflow-hidden relative">
                 {/* Message 1: System */}
                 <div className="flex gap-4 opacity-50">
                    <div className="w-8 h-8 rounded bg-white/10 flex items-center justify-center shrink-0">
                       <Cpu size={14} className="text-white" />
                    </div>
                    <div className="text-xs text-white/60 leading-relaxed">
                       System ready. Indexing "Financial_Q3.pdf"... <br/>
                       Found 12 data points. Ready for query.
                    </div>
                 </div>

                 {/* Message 2: User */}
                 <motion.div 
                    initial={{ opacity: 0, x: 20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 }}
                    className="flex flex-col items-end gap-2"
                 >
                    <div className="bg-white/10 text-white text-xs p-3 rounded-2xl rounded-tr-sm max-w-[80%]">
                       What is the total revenue reported for Q3, and does it exceed the projection?
                    </div>
                 </motion.div>

                 {/* Message 3: AI */}
                 <motion.div 
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: 1.5 }}
                    className="flex gap-4"
                 >
                    <div className="w-8 h-8 rounded bg-neon flex items-center justify-center shrink-0">
                       <span className="text-black font-bold text-xs">SP</span>
                    </div>
                    <div className="space-y-2 max-w-[85%]">
                       <div className="bg-neon/10 border border-neon/20 text-white text-xs p-3 rounded-2xl rounded-tl-sm leading-relaxed">
                          <p className="mb-2">Based on <span className="text-neon underline decoration-dotted cursor-help">Page 4, Table 2.1</span>:</p>
                          <p>Total Revenue for Q3 is <strong>$1,240,000</strong>.</p>
                          <p className="mt-2 text-white/70">Analysis: This exceeds the projected $1.1M by approximately 12.7%.</p>
                       </div>
                       <div className="flex gap-2">
                          <span className="text-[10px] bg-white/5 px-2 py-1 rounded text-white/40">Confidence: 98%</span>
                          <span className="text-[10px] bg-white/5 px-2 py-1 rounded text-white/40">Source: OCR</span>
                       </div>
                    </div>
                 </motion.div>

                 {/* Processing Indicator */}
                 <motion.div 
                    className="absolute bottom-6 left-6 flex items-center gap-2"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: [0, 1, 0] }}
                    transition={{ duration: 3, repeat: Infinity, delay: 4 }}
                 >
                    <div className="flex gap-1">
                       <span className="w-1 h-1 bg-neon rounded-full"></span>
                       <span className="w-1 h-1 bg-neon rounded-full"></span>
                       <span className="w-1 h-1 bg-neon rounded-full"></span>
                    </div>
                    <span className="text-[10px] text-neon uppercase">Thinking</span>
                 </motion.div>
              </div>

              {/* Input Area */}
              <div className="h-16 border-t border-white/10 bg-obsidian flex items-center px-6 gap-4">
                 <div className="w-6 h-6 rounded-full border border-white/20 flex items-center justify-center">
                    <span className="text-white/50 text-xs">+</span>
                 </div>
                 <div className="h-2 bg-white/10 rounded-full flex-1"></div>
                 <div className="w-6 h-6 text-neon">
                    <ArrowRightIcon />
                 </div>
              </div>

            </div>

            {/* Hardware Buttons */}
            <div className="absolute top-24 -right-[2px] w-[2px] h-10 bg-white/20 rounded-r-md"></div>
          </div>
        </div>
      </div>
    </section>
  );
};

const ArrowRightIcon = () => (
   <svg width="100%" height="100%" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" />
      <path d="m12 5 7 7-7 7" />
   </svg>
);