import React from 'react';

export const Manifesto = () => {
  return (
    <section id="manifesto" className="py-24 bg-neon text-obsidian clip-diagonal relative z-10 -mt-20 pt-32 pb-40">
      <div className="max-w-4xl mx-auto px-6 text-center">
        <h2 className="font-display text-4xl md:text-6xl font-bold uppercase mb-8 leading-none">
          Privacy is not a feature.<br/>It is the architecture.
        </h2>
        <p className="font-mono text-lg md:text-xl font-medium leading-relaxed max-w-2xl mx-auto">
          In an era of unchecked data harvesting, true intelligence requires isolation. 
          SilentPartner bridges the gap between chaotic files and structured knowledge, 
          without sending a single byte of your secrets to the cloud.
        </p>
      </div>
    </section>
  );
};