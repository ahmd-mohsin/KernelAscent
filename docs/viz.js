// KernelAscent visuals — white + red, small, centered, VERTICAL. Pure SVG+CSS, no deps.
// 1. Pipeline: 5 stacked task-INTERNAL schematics (model / harness / LoRA adapter / archive + loop-backs).
// 2. Causality DAG (bundled) and 3. scatter — recolored white/red, smaller.
const C = { red:"#c1121f", red2:"#e5484d", redwash:"#fdecec", ink:"#1a1a1a", mut:"#6b6b6b",
            grid:"#e4e4e7", paper:"#ffffff", wash:"#faf7f7", slate:"#2b3b52", gold:"#9a7b1f",
            green:"#2e7d5b", amber:"#b8860b" };
const outColor = m => m.rsi ? C.green : (m.wall_crossed ? C.amber : C.red);

// small helpers for the internal schematics
function box(x,y,w,h,label,sub,fill,stroke){
  let s=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="${fill||C.paper}" stroke="${stroke||C.ink}" stroke-width="1.1"/>`;
  s+=`<text x="${x+w/2}" y="${y+(sub?h/2-1:h/2+4)}" text-anchor="middle" font-size="11.5" font-weight="600" fill="${C.ink}">${label}</text>`;
  if(sub) s+=`<text x="${x+w/2}" y="${y+h/2+13}" text-anchor="middle" font-size="9" fill="${C.mut}">${sub}</text>`;
  return s;
}
function arr(x1,y1,x2,y2,col,dash){ // straight arrow
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col||C.ink}" stroke-width="1.3" marker-end="url(#tip)" ${dash?`stroke-dasharray="${dash}"`:""}/>`;
}
function loopBack(x1,y,x2,drop,col){ // animated red loop-back arrow under a row
  const d=`M${x1} ${y} C${x1} ${y+drop} ${x2} ${y+drop} ${x2} ${y+6}`;
  return `<path d="${d}" fill="none" stroke="${col}" stroke-width="1.3" marker-end="url(#tip-r)"/>`
       + `<path d="${d}" fill="none" class="edge-flow" marker-end="url(#tip-r)"/>`;
}

// ============================================================ 1. PIPELINE — vertical task internals
function renderTaskFlow(){
  const el=document.getElementById("viz-flow"); if(!el) return;
  const W=860, rowH=132, top=96, W0=150;
  const tasks=[
    {t:"T1 · Capability", note:"one-shot skill",
     boxes:[["prompt","task spec"],["model","frozen"],["kernel","ModelNew"],["grade","fp32 ✓ + speed"]], loop:null},
    {t:"T2 · Weight-RSI", note:"self-training compounds?",
     boxes:[["model","+LoRA"],["generate k","kernels"],["grade","keep correct"],["SFT adapter","update LoRA"]], loop:"weights update → next round"},
    {t:"T3 · Procedure-RSI", note:"weights frozen",
     boxes:[["procedure","prompt+strategies+archive"],["solve","frozen model"],["grade","verify"],["rewrite","edit procedure"]], loop:"improved procedure → next round"},
    {t:"T4 · Closed→Open", note:"closed edits open harness",
     boxes:[["closed researcher","fixed"],["edit harness","data · LoRA hp · curric."],["train trainee","open weights"],["eval","improved−frozen"]], loop:null},
    {t:"T5 · Self-play", note:"author co-evolution",
     boxes:[["author","propose harder"],["solver","solve"],["grade","gate hacks"],["both update","L − F"]], loop:"author + solver co-evolve"},
  ];
  const H = top + tasks.length*rowH + 20;
  let svg=`<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="RSI ladder — task internals">`;
  svg+=`<defs><marker id="tip" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 L9 5 L1 9 Z" fill="${C.ink}"/></marker>`
     + `<marker id="tip-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 L9 5 L1 9 Z" fill="${C.red}"/></marker></defs>`;
  svg+=`<text x="${W/2}" y="40" text-anchor="middle" class="stage-title">The KernelAscent RSI ladder — what each task does inside</text>`;
  svg+=`<text x="${W/2}" y="62" text-anchor="middle" class="body-label">one-shot skill → weight training loop → procedure edit loop → closed-drives-open → self-play co-evolution</text>`;
  tasks.forEach((tk,r)=>{
    const y0=top+r*rowH, bw=118, bh=46, gap=(W-W0-30-4*bw)/3, by=y0+34;
    svg+=`<text x="20" y="${y0+30}" class="stage-title" font-size="14" fill="${C.red}">${tk.t}</text>`;
    svg+=`<text x="20" y="${y0+48}" class="body-label">${tk.note}</text>`;
    const xs=[];
    tk.boxes.forEach((b,i)=>{ const x=W0+i*(bw+gap); xs.push(x);
      const fill = (i===0)?C.wash : (i===3&&tk.loop)?C.redwash : C.paper;
      svg+=box(x,by,bw,bh,b[0],b[1],fill, (i===3&&tk.loop)?C.red:C.ink);
      if(i<3) svg+=arr(x+bw,by+bh/2,x+bw+gap,by+bh/2, C.ink); });
    if(tk.loop){ // red animated loop-back from last box to first
      svg+=loopBack(xs[3]+bw/2, by+bh, xs[0]+bw/2, 30, C.red);
      svg+=`<text x="${(xs[0]+xs[3])/2+bw/2}" y="${by+bh+42}" text-anchor="middle" font-size="9.5" fill="${C.red}">${tk.loop}</text>`;
    } else {
      svg+=`<text x="${xs[3]+bw/2}" y="${by+bh+18}" text-anchor="middle" font-size="9.5" fill="${C.mut}">no weight/procedure loop</text>`;
    }
    if(r<tasks.length-1) svg+=`<line x1="20" y1="${y0+rowH-6}" x2="${W-20}" y2="${y0+rowH-6}" stroke="${C.grid}"/>`;
  });
  svg+=`</svg>`; el.innerHTML=svg;
}

// ============================================================ 2. CAUSALITY DAG (bundled, white/red)
function renderCausalityDAG(models){
  const el=document.getElementById("viz-dag"); if(!el) return;
  const W=820,H=460, gates=["SCALE","WALL","GRAD","DRIFT","RETAIN","DIVERSE","OUT"];
  const gx=i=>70+i*(W-120)/(gates.length-1), topY=96, botY=372;
  const stopGate=m=>!m.wall_crossed?1:(m.drift_total<0.05?2:(m.rsi?6:5));
  const B={}; models.forEach(m=>{const sg=stopGate(m),oc=m.rsi?"rsi":(m.wall_crossed?"flat":"wall");const k=sg+"|"+oc;(B[k]=B[k]||{sg,oc,n:0,drift:0,col:outColor(m)}).n++;B[k].drift+=(m.drift_total||0);});
  const bundles=Object.values(B).sort((a,b)=>a.sg-b.sg||(a.oc<b.oc?-1:1));
  const bySg={}; bundles.forEach(b=>(bySg[b.sg]=bySg[b.sg]||[]).push(b));
  let svg=`<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Causality DAG">`;
  svg+=`<text x="${W/2}" y="34" text-anchor="middle" class="stage-title">Where RSI breaks — internal gates</text>`;
  svg+=`<text x="${W/2}" y="54" text-anchor="middle" class="body-label">${models.length} runs bundled by where they stall · width ∝ Σ LoRA drift · n = runs · color = outcome</text>`;
  gates.forEach((g,i)=>{svg+=`<line x1="${gx(i)}" y1="${topY-10}" x2="${gx(i)}" y2="${botY+10}" stroke="${C.grid}"/><text x="${gx(i)}" y="${topY-18}" text-anchor="middle" class="gatehdr">${g}</text>`;});
  bundles.forEach(b=>{ const slots=bySg[b.sg],idx=slots.indexOf(b),yb=topY+(botY-topY)*((idx+1)/(slots.length+1));
    const w=Math.max(2,Math.min(26,2+b.drift*7));
    let d=`M${gx(0)} ${yb}`; for(let i=1;i<=b.sg;i++){const x=gx(i),px=gx(i-1);const y=(i===6&&b.oc==="rsi")?topY+40:yb;d+=` C${(px+x)/2} ${yb} ${(px+x)/2} ${y} ${x} ${y}`;}
    const op=b.oc==="rsi"?0.85:(b.oc==="flat"?0.55:0.4);
    svg+=`<path d="${d}" fill="none" stroke="${b.col}" stroke-width="${w.toFixed(1)}" stroke-opacity="${op}" stroke-linecap="round"/>`;
    const ex=gx(b.sg),ey=(b.sg===6&&b.oc==="rsi")?topY+40:yb;
    svg+=`<circle cx="${ex}" cy="${ey}" r="${(w/2+5).toFixed(1)}" fill="${C.paper}" stroke="${b.col}" stroke-width="1.4" stroke-opacity="${op}"/><text x="${ex}" y="${ey+3.5}" text-anchor="middle" class="mono" font-size="10" fill="${b.col}">${b.n}</text>`; });
  const leg=[["RSI",C.green],["flat",C.amber],["stuck",C.red]]; leg.forEach(([t,c],i)=>{const lx=70+i*150;svg+=`<circle cx="${lx}" cy="${botY+34}" r="5" fill="${c}"/><text x="${lx+12}" y="${botY+38}" class="body-label">${t}</text>`;});
  svg+=`</svg>`; el.innerHTML=svg;
}

// ============================================================ 3. SCATTER (white/red, center dot + halo)
function renderScaleGain(models){
  const el=document.getElementById("viz-scatter"); if(!el) return;
  const W=820,H=520,mL=64,mR=120,mT=120,mB=64, px=W-mR,py=H-mB;
  const xs=models.map(m=>Math.log(m.size_b||2)); const xlo=Math.min(...xs)-.15,xhi=Math.max(...xs)+.15;
  const gains=models.map(m=>m.C_held_gain||0); const ylo=Math.min(-.05,...gains),yhi=Math.max(.05,...gains);
  const X=v=>mL+(px-mL)*(Math.log(v)-xlo)/(xhi-xlo); const Y=v=>py-(py-mT)*(v-ylo)/(yhi-ylo);
  let svg=`<svg class="viz-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Scale vs RSI gain">`;
  svg+=`<text x="${W/2}" y="34" text-anchor="middle" class="stage-title">Scale vs held-out RSI gain — an inverted-U</text>`;
  svg+=`<text x="${W/2}" y="54" text-anchor="middle" class="body-label">dot = model · halo area ∝ LoRA drift · gain peaks mid-scale; ≥9B drifts most yet gains least</text>`;
  const wallX=X(2); svg+=`<rect x="${mL}" y="${mT}" width="${wallX-mL}" height="${py-mT}" fill="${C.redwash}" opacity=".7"/><text x="${(mL+wallX)/2}" y="${mT+18}" text-anchor="middle" class="body-label" font-style="italic">sub-2B wall</text>`;
  svg+=`<line x1="${mL}" y1="${py}" x2="${px}" y2="${py}" stroke="${C.grid}"/><line x1="${mL}" y1="${mT}" x2="${mL}" y2="${py}" stroke="${C.grid}"/>`;
  svg+=`<line x1="${mL}" y1="${Y(0)}" x2="${px}" y2="${Y(0)}" stroke="${C.grid}" stroke-dasharray="3 5"/><text x="${px+5}" y="${Y(0)+3}" class="axis-label">0</text>`;
  [0.5,1,2,3,7,14,32].forEach(s=>{if(s>=Math.exp(xlo)&&s<=Math.exp(xhi))svg+=`<text x="${X(s)}" y="${py+20}" text-anchor="middle" class="axis-label mono">${s}B</text>`;});
  svg+=`<text x="${(mL+px)/2}" y="${py+42}" text-anchor="middle" class="axis-label">model size (log)</text>`;
  svg+=`<text x="18" y="${(mT+py)/2}" text-anchor="middle" class="axis-label" transform="rotate(-90 18 ${(mT+py)/2})">held-out gain</text>`;
  [[0,2,"<2B"],[2,8,"2–8B"],[8,99,"≥9B"]].forEach(([a,b,lab])=>{const inb=models.filter(m=>(m.size_b||2)>=a&&(m.size_b||2)<b);if(!inb.length)return;const fr=inb.filter(m=>m.rsi).length/inb.length;const x0=X(Math.max(a,Math.exp(xlo))),x1=X(Math.min(b,Math.exp(xhi)));const bh=54*fr;svg+=`<rect x="${x0+3}" y="${mT-16-bh}" width="${x1-x0-6}" height="${bh}" fill="${C.red}" opacity=".4" rx="2"/><text x="${(x0+x1)/2}" y="${mT-20-bh}" text-anchor="middle" class="body-label">${(fr*100).toFixed(0)}%</text><text x="${(x0+x1)/2}" y="${mT-4}" text-anchor="middle" class="axis-label mono">${lab}</text>`;});
  models.forEach(m=>{const col=outColor(m);const r=Math.max(3,Math.sqrt((m.drift_total||0)*900/Math.PI));const x=X(m.size_b||2).toFixed(1),y=Y(m.C_held_gain||0).toFixed(1);svg+=`<circle cx="${x}" cy="${y}" r="${r.toFixed(1)}" fill="${col}" fill-opacity=".10" stroke="${col}" stroke-opacity=".35"/><circle cx="${x}" cy="${y}" r="2.2" fill="${col}"/>`;});
  const leg=[["RSI",C.green],["flat",C.amber],["stuck",C.red]]; leg.forEach(([t,c],i)=>{svg+=`<circle cx="${px+34}" cy="${mT+30+i*22}" r="5" fill="${c}"/><text x="${px+46}" y="${mT+34+i*22}" class="body-label">${t}</text>`;});
  svg+=`</svg>`; el.innerHTML=svg;
}

(function init(){
  fetch("data/mech_analysis.json").then(r=>r.json()).then(d=>{
    const models=(d.models||[]).filter(m=>typeof m.size_b==="number");
    try{renderTaskFlow();}catch(e){console.error("flow",e);}
    try{renderCausalityDAG(models);}catch(e){console.error("dag",e);}
    try{renderScaleGain(models);}catch(e){console.error("scatter",e);}
  }).catch(e=>console.error("viz load",e));
})();
