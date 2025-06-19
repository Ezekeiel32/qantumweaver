import { redirect } from 'next/navigation';

export default function Home() {
  return (
    <div className="neon-card glass min-h-screen flex flex-col items-center justify-center">
      <h1 className="neon-heading text-5xl mb-8">Quantum Weaver</h1>
      {/* ...rest of your content... */}
    </div>
  );
}
