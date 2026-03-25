"""
live/dashboard.py
-----------------
Local web dashboard for the swing trading system.
v2: Auto-refresh, intraday charts, market pulse, live prices.

Usage:
    python live/dashboard.py          # Start on http://localhost:5050
    python main.py dashboard          # Via main entry point
"""

import sys
import json
import webbrowser
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from loguru import logger

sys.path.insert(0, ".")

try:
    from flask import Flask, jsonify, render_template_string, request
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)

MARKET_PULSE_CACHE = {"data": None, "timestamp": None}


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swing Trader v3</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #06080f; --bg-card: #0c1018; --border: #1a2035; --border-glow: #1e3a5f;
    --text: #c9d1d9; --muted: #4a5568; --dim: #2d3748;
    --green: #00e676; --green-dim: rgba(0,230,118,0.12);
    --red: #ff5252; --red-dim: rgba(255,82,82,0.12);
    --blue: #448aff; --blue-dim: rgba(68,138,255,0.12);
    --purple: #b388ff; --purple-dim: rgba(179,136,255,0.12);
    --yellow: #ffd740; --yellow-dim: rgba(255,215,64,0.12);
    --cyan: #18ffff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); overflow-x: hidden; }
  .mono { font-family: 'JetBrains Mono', monospace; }

  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 20px; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #0a0e18, var(--bg));
    position: sticky; top: 0; z-index: 100;
  }
  .logo { display: flex; align-items: baseline; gap: 10px; }
  .logo h1 { font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 800; letter-spacing: 0.15em; color: #fff; }
  .logo .ver { font-size: 8px; color: var(--green); font-weight: 700; background: var(--green-dim); padding: 2px 5px; border-radius: 3px; letter-spacing: 0.1em; }
  .header-right { display: flex; align-items: center; gap: 14px; }
  .mkt-status { display: flex; align-items: center; gap: 5px; font-size: 9px; font-weight: 700; letter-spacing: 0.08em; }
  .dot { width: 6px; height: 6px; border-radius: 50%; animation: pulse 2s ease infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
  .dot-g { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot-r { background: var(--red); box-shadow: 0 0 6px var(--red); }
  .dot-y { background: var(--yellow); box-shadow: 0 0 6px var(--yellow); }
  .regime-pill { padding: 3px 9px; border-radius: 4px; font-size: 9px; font-weight: 800; letter-spacing: 0.08em; }
  .regime-ok { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,230,118,0.2); }
  .regime-bad { background: var(--red-dim); color: var(--red); border: 1px solid rgba(255,82,82,0.2); }
  .clock { font-size: 10px; color: var(--muted); }
  .ref-dot { font-size: 8px; color: var(--dim); transition: color 0.3s; }
  .ref-dot.on { color: var(--cyan); }

  .main { padding: 12px 16px; display: flex; flex-direction: column; gap: 10px; }
  .row { display: flex; gap: 10px; }
  .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; transition: border-color 0.3s; }
  .card:hover { border-color: var(--border-glow); }
  .card-t { font-size: 8px; font-weight: 800; letter-spacing: 0.14em; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; }

  .kpi-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
  .kpi { padding: 10px 12px; }
  .kpi .v { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 800; line-height: 1.1; }
  .kpi .l { font-size: 8px; color: var(--muted); letter-spacing: 0.1em; font-weight: 700; text-transform: uppercase; margin-bottom: 3px; }
  .kpi .s { font-size: 9px; color: var(--muted); margin-top: 2px; }

  .green { color: var(--green); } .red { color: var(--red); } .blue { color: var(--blue); }
  .purple { color: var(--purple); } .yellow { color: var(--yellow); } .muted { color: var(--muted); }

  .pulse-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .pulse-item { background: rgba(255,255,255,0.02); border-radius: 5px; padding: 8px; text-align: center; }
  .pulse-item .tk { font-size: 9px; font-weight: 800; color: var(--muted); letter-spacing: 0.08em; }
  .pulse-item .pr { font-size: 15px; font-weight: 800; margin: 2px 0; }
  .pulse-item .ch { font-size: 10px; font-weight: 700; }

  .chart-area { width: 100%; height: 150px; }
  .chart-area svg { width: 100%; height: 100%; }
  .intra-chart { height: 120px; }

  table { width: 100%; border-collapse: collapse; font-size: 10px; }
  th { text-align: left; font-size: 8px; color: var(--muted); letter-spacing: 0.08em; font-weight: 800; padding: 6px 8px; border-bottom: 1px solid var(--border); text-transform: uppercase; }
  td { padding: 7px 8px; border-bottom: 1px solid rgba(255,255,255,0.02); }
  tr:hover td { background: rgba(255,255,255,0.015); }

  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 8px; font-weight: 800; letter-spacing: 0.06em; }
  .badge-hold { color: var(--dim); } .badge-sell { background: var(--yellow-dim); color: var(--yellow); }
  .badge-hedge { background: var(--purple-dim); color: var(--purple); } .badge-buy { background: var(--green-dim); color: var(--green); }

  .prox-bar { width: 70px; height: 3px; background: #1a1a2e; border-radius: 2px; position: relative; display: inline-block; vertical-align: middle; }
  .prox-mk { width: 2px; height: 8px; border-radius: 1px; position: absolute; top: -2.5px; }

  .perf-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .perf-stat .pl { font-size: 8px; color: var(--muted); letter-spacing: 0.08em; font-weight: 700; }
  .perf-stat .pv { font-size: 16px; font-weight: 800; margin-top: 1px; }

  .footer { padding: 6px 20px; border-top: 1px solid var(--border); font-size: 8px; color: var(--dim); display: flex; justify-content: space-between; }
  .footer .tag { padding: 1px 4px; border-radius: 2px; font-weight: 700; background: rgba(255,255,255,0.03); letter-spacing: 0.04em; margin-right: 8px; }

  .flash-g { animation: fG 0.5s ease; } .flash-r { animation: fR 0.5s ease; }
  @keyframes fG { 0% { background: var(--green-dim); } 100% { background: transparent; } }
  @keyframes fR { 0% { background: var(--red-dim); } 100% { background: transparent; } }

  ::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: var(--bg); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
</head>
<body>
<div class="header">
  <div class="logo"><h1>SWING TRADER</h1><span class="ver mono">V3</span></div>
  <div id="rpill" class="regime-pill regime-bad mono">LOADING</div>
  <div class="header-right">
    <div class="mkt-status mono" id="mkt-st"><span class="dot dot-y"></span> —</div>
    <div class="clock mono" id="clk"></div>
    <div class="ref-dot mono" id="ref">●</div>
  </div>
</div>
<div class="main">
  <div class="kpi-row">
    <div class="card kpi"><div class="l mono">EQUITY</div><div class="v mono" id="k-eq">—</div><div class="s mono" id="k-ret">—</div></div>
    <div class="card kpi"><div class="l mono">CASH</div><div class="v mono" id="k-cash">—</div><div class="s mono" id="k-inv">—</div></div>
    <div class="card kpi"><div class="l mono">TODAY P&L</div><div class="v mono" id="k-daily">—</div><div class="s mono" id="k-cum">—</div></div>
    <div class="card kpi"><div class="l mono">POSITIONS</div><div class="v mono" id="k-pos">—</div><div class="s mono" id="k-tr">—</div></div>
    <div class="card kpi"><div class="l mono">VIX</div><div class="v mono" id="k-vix">—</div><div class="s mono" id="k-br">—</div></div>
  </div>
  <div class="row">
    <div class="card" style="flex:1.4">
      <div class="card-t mono">MARKET PULSE</div>
      <div class="pulse-grid">
        <div class="pulse-item"><div class="tk mono">SPY</div><div class="pr mono" id="p-spy-p">—</div><div class="ch mono" id="p-spy-c">—</div></div>
        <div class="pulse-item"><div class="tk mono">QQQ</div><div class="pr mono" id="p-qqq-p">—</div><div class="ch mono" id="p-qqq-c">—</div></div>
        <div class="pulse-item"><div class="tk mono">IWM</div><div class="pr mono" id="p-iwm-p">—</div><div class="ch mono" id="p-iwm-c">—</div></div>
        <div class="pulse-item"><div class="tk mono">VIX</div><div class="pr mono" id="p-vix-p">—</div><div class="ch mono" id="p-vix-c">—</div></div>
      </div>
      <div style="margin-top:10px"><div class="card-t mono">INTRADAY P&L</div><div class="intra-chart" id="intra-ch"><div style="color:var(--dim);text-align:center;padding:30px;font-size:10px">Building throughout the day...</div></div></div>
    </div>
    <div class="card" style="flex:1">
      <div class="card-t mono">EQUITY CURVE</div>
      <div class="chart-area" id="eq-ch"><div style="color:var(--dim);text-align:center;padding:50px;font-size:10px">Loading...</div></div>
      <div style="margin-top:12px"><div class="card-t mono">TRADE STATS</div>
        <div class="perf-grid">
          <div class="perf-stat"><div class="pl mono">WIN RATE</div><div class="pv mono green" id="s-wr">—</div></div>
          <div class="perf-stat"><div class="pl mono">PROFIT FACTOR</div><div class="pv mono green" id="s-pf">—</div></div>
          <div class="perf-stat"><div class="pl mono">TOTAL P&L</div><div class="pv mono" id="s-pnl">—</div></div>
          <div class="perf-stat"><div class="pl mono">AVG RETURN</div><div class="pv mono" id="s-avg">—</div></div>
        </div>
      </div>
    </div>
  </div>
  <div class="row">
    <div class="card" style="flex:1.3">
      <div class="card-t mono">OPEN POSITIONS</div>
      <table><thead><tr><th>TICKER</th><th>ENTRY</th><th>CURRENT</th><th>P&L</th><th>DAYS</th><th>PROXIMITY</th><th>ACTION</th></tr></thead>
      <tbody id="pos-body"></tbody></table>
      <div id="no-pos" style="color:var(--dim);font-size:10px;padding:16px;text-align:center">Loading...</div>
    </div>
    <div class="card" style="flex:0.7">
      <div class="card-t mono">RECENT TRADES</div>
      <table><thead><tr><th>TICKER</th><th>P&L</th><th>DAYS</th><th>REASON</th></tr></thead>
      <tbody id="hist-body"></tbody></table>
      <div id="no-hist" style="color:var(--dim);font-size:10px;padding:16px;text-align:center">Loading...</div>
    </div>
  </div>
</div>
<div class="footer">
  <div class="mono"><span class="tag">LightGBM v3</span><span class="tag">3-Day Target</span><span class="tag">Conservative</span> Walk-Forward · 25 Folds · AUC 0.599</div>
  <span class="mono" id="upd">—</span>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=(n,d=0)=>'$'+(n||0).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=n=>(n>=0?'+':'')+(n||0).toFixed(1)+'%';
const pct2=n=>(n>=0?'+':'')+(n||0).toFixed(2)+'%';

let prevEq=null, intra=[];

function isMktHrs(){const n=new Date(),e=new Date(n.toLocaleString("en-US",{timeZone:"America/New_York"})),h=e.getHours(),m=e.getMinutes(),d=e.getDay();if(d===0||d===6)return false;const t=h*60+m;return t>=570&&t<=960;}
function mktText(){const n=new Date(),e=new Date(n.toLocaleString("en-US",{timeZone:"America/New_York"})),h=e.getHours(),m=e.getMinutes(),d=e.getDay();if(d===0||d===6)return{t:'WEEKEND',c:'dot-y'};const t=h*60+m;if(t<570)return{t:'PRE-MARKET',c:'dot-y'};if(t<=960)return{t:'MARKET OPEN',c:'dot-g'};return{t:'AFTER HOURS',c:'dot-y'};}

function loadAll(){
  const r=$('ref');r.classList.add('on');setTimeout(()=>r.classList.remove('on'),800);
  fetch('/api/status').then(r=>r.json()).then(d=>{
    // KPIs
    const p=d.portfolio,eq=$('k-eq'),nv=p.total_equity;
    eq.textContent=fmt(nv);eq.className='v mono '+(p.total_return_pct>=0?'green':'red');
    if(prevEq!==null&&nv!==prevEq){eq.parentElement.classList.add(nv>prevEq?'flash-g':'flash-r');setTimeout(()=>eq.parentElement.classList.remove('flash-g','flash-r'),600);}
    prevEq=nv;
    $('k-ret').textContent=pct(p.total_return_pct)+' total';
    $('k-cash').textContent=fmt(p.cash);$('k-cash').className='v mono';
    $('k-inv').textContent=fmt(p.invested)+' invested';
    $('k-daily').textContent=fmt(p.daily_pnl);$('k-daily').className='v mono '+(p.daily_pnl>=0?'green':'red');
    $('k-cum').textContent=fmt(p.cumulative_pnl)+' cumulative';
    $('k-pos').textContent=d.positions.length+'/'+(d.config.max_positions||5);
    $('k-pos').className='v mono '+(d.positions.length>0?'blue':'');
    $('k-tr').textContent=(d.performance.total_trades||0)+' trades';
    // Regime
    const rp=$('rpill');
    if(d.regime.regime_ok){rp.className='regime-pill regime-ok mono';rp.textContent='FAVORABLE';}
    else{rp.className='regime-pill regime-bad mono';rp.textContent='UNFAVORABLE';}
    $('k-vix').textContent=d.regime.vix||'—';
    $('k-vix').className='v mono '+((d.regime.vix||0)>25?'red':(d.regime.vix||0)>18?'yellow':'green');
    $('k-br').textContent=(d.regime.pct_above||0)+'% > SMA50';
    // Positions
    const pb=$('pos-body'),np=$('no-pos');
    if(d.positions.length>0){np.style.display='none';pb.innerHTML=d.positions.map(p=>{
      const pnl=p.unrealized_pct||0,cls=pnl>=0?'green':'red',days=p.days_held||'?',act=p.action||'HOLD';
      let badge;if(p.is_hedge)badge='<span class="badge badge-hedge mono">HEDGE</span>';
      else if(act!=='HOLD')badge='<span class="badge badge-sell mono">'+act.replace('SELL_','')+'</span>';
      else badge='<span class="badge badge-hold mono">HOLD</span>';
      let prox='';if(p.entry_price&&p.stop_price&&p.target_price&&p.current_price){
        const rng=p.target_price-p.stop_price,pos=rng>0?((p.current_price-p.stop_price)/rng*100):50,cl=Math.max(0,Math.min(100,pos));
        prox='<div class="prox-bar"><div class="prox-mk" style="left:0;background:var(--red)"></div><div class="prox-mk" style="left:'+cl+'%;background:#fff"></div><div class="prox-mk" style="left:100%;background:var(--green)"></div></div>';}
      return '<tr><td class="mono" style="font-weight:800;color:#fff">'+p.ticker+'</td><td class="mono muted">'+fmt(p.entry_price,2)+'</td><td class="mono">'+fmt(p.current_price,2)+'</td><td class="mono '+cls+'">'+pct(pnl)+'</td><td class="mono muted">'+days+'d</td><td>'+prox+'</td><td>'+badge+'</td></tr>';
    }).join('');}else{np.style.display='block';np.textContent='No open positions — waiting for favorable regime';pb.innerHTML='';}
    // History
    const hb=$('hist-body'),nh=$('no-hist');
    if(d.history.length>0){nh.style.display='none';hb.innerHTML=d.history.slice(0,12).map(t=>{
      const pnl=t.pnl_pct||0,cls=pnl>=0?'green':'red';
      return '<tr><td class="mono" style="font-weight:700">'+t.ticker+'</td><td class="mono '+cls+'">'+pct(pnl*100)+'</td><td class="mono muted">'+(t.hold_days||'—')+'d</td><td class="mono muted">'+(t.exit_reason||'')+'</td></tr>';
    }).join('');}else{nh.style.display='block';nh.textContent='No closed trades yet';hb.innerHTML='';}
    // Performance
    if(d.performance.total_trades>0){$('s-wr').textContent=(d.performance.win_rate||0).toFixed(0)+'%';$('s-pf').textContent=(d.performance.profit_factor||0).toFixed(2);
      const pe=$('s-pnl');pe.textContent=fmt(d.performance.total_pnl);pe.className='pv mono '+((d.performance.total_pnl||0)>=0?'green':'red');$('s-avg').textContent=pct(d.performance.avg_pnl_pct||0);}
    // Equity chart
    drawEqChart(d.equity_curve);
    // Intraday
    if(nv>0){intra.push({t:new Date(),e:nv});const cut=Date.now()-8*3600000;intra=intra.filter(p=>p.t.getTime()>cut);drawIntra();}
    $('upd').textContent=new Date().toLocaleTimeString();
  }).catch(e=>console.error(e));
  fetch('/api/pulse').then(r=>r.json()).then(d=>{
    ['spy','qqq','iwm','vix'].forEach(s=>{if(!d[s])return;const i=d[s];
      $('p-'+s+'-p').textContent=i.price?i.price.toFixed(2):'—';
      if(i.change_pct!==undefined){const c=i.change_pct;$('p-'+s+'-c').textContent=pct2(c);
        $('p-'+s+'-c').className='ch mono '+(s==='vix'?(c>0?'red':'green'):(c>=0?'green':'red'));
        $('p-'+s+'-p').className='pr mono '+(s==='vix'?(c>0?'red':'green'):(c>=0?'green':'red'));}});
  }).catch(e=>console.error(e));
}

function drawEqChart(data){
  const el=$('eq-ch');if(!data||data.length<2){el.innerHTML='<div style="color:var(--dim);text-align:center;padding:50px;font-size:10px">Equity curve builds with trades</div>';return;}
  const W=500,H=140,P=36,vals=data.map(d=>d.total_equity),mn=Math.min(...vals)*0.98,mx=Math.max(...vals)*1.02,rng=mx-mn||1;
  const tX=i=>P+(i/Math.max(vals.length-1,1))*(W-P-8),tY=v=>6+(1-(v-mn)/rng)*(H-20);
  const pts=vals.map((v,i)=>tX(i)+','+tY(v)).join(' '),fill=tX(0)+','+H+' '+pts+' '+tX(vals.length-1)+','+H;
  let g='';[.25,.5,.75].forEach(f=>{const v=mn+f*rng,y=tY(v);g+='<line x1="'+P+'" y1="'+y+'" x2="'+(W-8)+'" y2="'+y+'" stroke="#1a2035" stroke-width="0.5"/>';g+='<text x="'+(P-3)+'" y="'+(y+3)+'" text-anchor="end" fill="#4a5568" font-size="7" font-family="JetBrains Mono">$'+(v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0))+'</text>';});
  const up=vals[vals.length-1]>=vals[0],col=up?'#00e676':'#ff5252';
  el.innerHTML='<svg viewBox="0 0 '+W+' '+H+'">'+g+'<defs><linearGradient id="eg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="'+col+'" stop-opacity="0.15"/><stop offset="100%" stop-color="'+col+'" stop-opacity="0"/></linearGradient></defs><polygon fill="url(#eg)" points="'+fill+'"/><polyline fill="none" stroke="'+col+'" stroke-width="1.5" points="'+pts+'"/><circle cx="'+tX(vals.length-1)+'" cy="'+tY(vals[vals.length-1])+'" r="2.5" fill="'+col+'"/></svg>';
}

function drawIntra(){
  const el=$('intra-ch');if(intra.length<2){el.innerHTML='<div style="color:var(--dim);text-align:center;padding:30px;font-size:10px">Intraday P&L builds throughout the day</div>';return;}
  const W=500,H=100,P=36,base=intra[0].e,pnls=intra.map(d=>d.e-base);
  const mn=Math.min(0,...pnls),mx=Math.max(0,...pnls),rng=(mx-mn)||1;
  const tX=i=>P+(i/Math.max(pnls.length-1,1))*(W-P-8),tY=v=>6+(1-(v-mn)/rng)*(H-18);
  const zY=tY(0),pts=pnls.map((v,i)=>tX(i)+','+tY(v)).join(' '),latest=pnls[pnls.length-1];
  const col=latest>=0?'#00e676':'#ff5252',fill=tX(0)+','+zY+' '+pts+' '+tX(pnls.length-1)+','+zY;
  let tl='';[0,.5,1].forEach(f=>{const idx=Math.floor(f*(intra.length-1)),t=intra[idx].t,lb=t.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit',hour12:true});tl+='<text x="'+tX(idx)+'" y="'+(H-1)+'" text-anchor="middle" fill="#4a5568" font-size="6" font-family="JetBrains Mono">'+lb+'</text>';});
  el.innerHTML='<svg viewBox="0 0 '+W+' '+H+'"><line x1="'+P+'" y1="'+zY+'" x2="'+(W-8)+'" y2="'+zY+'" stroke="#2d3748" stroke-width="0.5" stroke-dasharray="3,3"/>'+tl+'<defs><linearGradient id="ig" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="'+col+'" stop-opacity="0.1"/><stop offset="100%" stop-color="'+col+'" stop-opacity="0"/></linearGradient></defs><polygon fill="url(#ig)" points="'+fill+'"/><polyline fill="none" stroke="'+col+'" stroke-width="1.5" points="'+pts+'"/><circle cx="'+tX(pnls.length-1)+'" cy="'+tY(latest)+'" r="2" fill="'+col+'"/><text x="'+(tX(pnls.length-1)+6)+'" y="'+(tY(latest)+3)+'" fill="'+col+'" font-size="9" font-weight="800" font-family="JetBrains Mono">'+(latest>=0?'+':'')+latest.toFixed(2)+'</text></svg>';
}

function updClk(){$('clk').textContent=new Date().toLocaleTimeString('en-US',{hour12:false});const m=mktText();$('mkt-st').innerHTML='<span class="dot '+m.c+'"></span> '+m.t;}
setInterval(updClk,1000);updClk();

// Auto-refresh: 30s during market, 5min after
loadAll();
(function loop(){setTimeout(()=>{loadAll();loop();},isMktHrs()?30000:300000);})();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    """Returns complete system status as JSON."""
    import yaml
    import pandas as pd
    from live.positions import (
        get_open_positions, update_positions_with_prices,
        get_trade_history, get_performance_summary,
        get_portfolio_summary, get_equity_curve, initialize_portfolio,
    )

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    initialize_portfolio(config["backtest"]["initial_capital"])

    portfolio = get_portfolio_summary()
    if not portfolio:
        portfolio = {
            "cash": config["backtest"]["initial_capital"],
            "invested": 0, "total_equity": config["backtest"]["initial_capital"],
            "initial_equity": config["backtest"]["initial_capital"],
            "total_return_pct": 0, "cumulative_pnl": 0, "daily_pnl": 0,
            "n_positions": 0, "n_snapshots": 0,
        }

    # Positions with live prices
    positions = get_open_positions()
    if positions:
        try:
            import yfinance as yf
            tickers = [p["ticker"] for p in positions]
            price_data = yf.download(tickers, period="2d", auto_adjust=True,
                                      progress=False, threads=True)
            if not price_data.empty:
                if len(tickers) == 1:
                    close = price_data["Close"]
                    if len(close) > 0:
                        positions[0]["current_price"] = float(close.iloc[-1])
                else:
                    close = price_data["Close"]
                    for p in positions:
                        if p["ticker"] in close.columns:
                            val = close[p["ticker"]].iloc[-1]
                            if pd.notna(val):
                                p["current_price"] = float(val)
        except Exception as e:
            logger.warning(f"Live price fetch failed: {e}")

    invested = 0
    for p in positions:
        p["current_price"] = p.get("current_price", p["entry_price"])
        p["unrealized_pct"] = (p["current_price"] / p["entry_price"] - 1) * 100 if p["entry_price"] else 0
        p["is_hedge"] = p["ticker"] in ("SH", "SDS", "SPXU", "PSQ", "DOG", "SQQQ")
        invested += p.get("size_usd", 0) * (p["current_price"] / p["entry_price"]) if p["entry_price"] > 0 else 0
        if p.get("entry_date"):
            try:
                entry = datetime.strptime(str(p["entry_date"]), "%Y-%m-%d").date()
                p["days_held"] = (date.today() - entry).days
            except Exception:
                p["days_held"] = 0

    portfolio["invested"] = invested

    history_df = get_trade_history()
    history = history_df.to_dict("records") if not history_df.empty else []
    performance = get_performance_summary()

    eq_df = get_equity_curve()
    equity_curve = []
    if not eq_df.empty:
        equity_curve = [
            {"date": str(r["date"].date()), "total_equity": r["total_equity"]}
            for _, r in eq_df.iterrows()
        ]

    regime = {"regime_ok": False, "pct_above": 0, "vix": None}
    try:
        from live.regime import get_todays_vix_context
        ctx = get_todays_vix_context()
        regime["vix"] = round(ctx.get("vix_current", 0), 1)
        regime["vix_regime"] = ctx.get("vix_regime", "UNKNOWN")
        regime["threshold"] = ctx.get("threshold", 0.50)
    except Exception:
        pass

    try:
        import yfinance as yf
        from data.universe import get_sp500_tickers, _clean_ticker_list, FALLBACK_TICKERS
        sample_tickers = _clean_ticker_list(FALLBACK_TICKERS)[:50]
        raw = yf.download(sample_tickers, period="60d", auto_adjust=True,
                          progress=False, threads=True)
        if not raw.empty:
            close = raw["Close"]
            sma50 = close.rolling(50).mean()
            latest_close = close.iloc[-1]
            latest_sma = sma50.iloc[-1]
            above = (latest_close > latest_sma).sum()
            total = latest_close.notna().sum()
            pct_above = (above / total * 100) if total > 0 else 0
            regime["pct_above"] = int(round(pct_above))
            threshold = regime.get("threshold", 0.50)
            regime["regime_ok"] = bool((pct_above / 100) >= threshold)
    except Exception as e:
        logger.warning(f"Breadth calculation failed: {e}")

    return jsonify({
        "portfolio": portfolio,
        "positions": positions,
        "history": history[:20],
        "performance": performance,
        "equity_curve": equity_curve,
        "regime": regime,
        "config": {
            "max_positions": config["universe"]["max_positions"],
            "initial_capital": config["backtest"]["initial_capital"],
        },
    })


@app.route("/api/pulse")
def api_pulse():
    """Returns market pulse: SPY, QQQ, IWM, VIX with daily changes."""
    import time
    now = time.time()
    if (MARKET_PULSE_CACHE["data"] is not None and
        MARKET_PULSE_CACHE["timestamp"] and
        now - MARKET_PULSE_CACHE["timestamp"] < 20):
        return jsonify(MARKET_PULSE_CACHE["data"])

    result = {}
    try:
        import yfinance as yf
        tickers = {"SPY": "spy", "QQQ": "qqq", "IWM": "iwm", "^VIX": "vix"}
        data = yf.download(list(tickers.keys()), period="2d", auto_adjust=True,
                            progress=False, threads=True)
        if not data.empty:
            close = data["Close"]
            for yf_ticker, key in tickers.items():
                if yf_ticker in close.columns and len(close) >= 2:
                    today_price = float(close[yf_ticker].iloc[-1])
                    prev_price = float(close[yf_ticker].iloc[-2])
                    change_pct = (today_price / prev_price - 1) * 100 if prev_price > 0 else 0
                    result[key] = {"price": today_price, "prev_close": prev_price, "change_pct": change_pct}
    except Exception as e:
        logger.warning(f"Pulse fetch failed: {e}")

    MARKET_PULSE_CACHE["data"] = result
    MARKET_PULSE_CACHE["timestamp"] = now
    return jsonify(result)


def run_dashboard(port: int = 5050):
    """Starts the dashboard web server."""
    logger.info(f"Starting dashboard on http://localhost:{port}")
    def open_browser():
        import time; time.sleep(1.5); webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    run_dashboard()
