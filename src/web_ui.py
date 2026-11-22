"""Single-page demo UI served at GET /. No external assets; inline CSS/JS only."""

CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Toxic Comment Classifier</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--card:#f5f5f7;--line:#d8d8de;--accent:#2563eb;
--c0:#16a34a;--c1:#ca8a04;--c2:#ea580c;--c3:#dc2626}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--fg:#ececf0;--muted:#9a9aa6;--card:#1b1e26;--line:#2d313c;--accent:#5b8cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:24px 16px}h1{font-size:1.4rem;margin:0 0 4px}
p.sub{color:var(--muted);margin:0 0 16px}
textarea,input{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
textarea{min-height:120px;resize:vertical}.row{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}
button{padding:9px 16px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;cursor:pointer}
button.ghost{background:var(--card);color:var(--fg);border:1px solid var(--line)}button:disabled{opacity:.5}
#result{margin-top:16px;padding:16px;background:var(--card);border-radius:10px;display:none}
.bar{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:.9rem}.bar span.n{width:150px}
.track{flex:1;height:12px;background:var(--line);border-radius:6px;overflow:hidden}.fill{height:100%}
#err{color:#dc2626;margin-top:10px}small{color:var(--muted)}
</style></head><body><main>
<h1>Toxic Comment Classifier</h1>
<p class="sub">4-class model (neutral, mildly problematic, moderately toxic, severely toxic). Demo only; not for automated moderation decisions.</p>
<textarea id="t" maxlength="20000" placeholder="Type or paste a comment..."></textarea>
<div class="row">
<button id="go">Classify</button>
<button class="ghost" data-ex="Thanks for the thoughtful article, I learned a lot.">Example: friendly</button>
<button class="ghost" data-ex="You are a complete idiot and nobody wants you here.">Example: insult</button>
</div>
<input id="k" type="password" placeholder="API key (only if the server requires one)" autocomplete="off">
<div id="err" role="alert"></div>
<div id="result"><strong id="lab"></strong><div id="bars"></div></div>
<p><small>The text is sent to this server for prediction. The server does not log comment text. The API key is kept in memory only.</small></p>
</main>
<script>
const COLORS={neutral:'var(--c0)',mildly_problematic:'var(--c1)',moderately_toxic:'var(--c2)',severely_toxic:'var(--c3)'};
const $=id=>document.getElementById(id);
document.querySelectorAll('[data-ex]').forEach(b=>b.onclick=()=>{$('t').value=b.dataset.ex;});
$('go').onclick=async()=>{
  const text=$('t').value.trim(); $('err').textContent='';
  if(!text){$('err').textContent='Enter a comment first.';return;}
  $('go').disabled=true;
  try{
    const headers={'Content-Type':'application/json'}; if($('k').value) headers['X-API-Key']=$('k').value;
    const r=await fetch('/predict',{method:'POST',headers,body:JSON.stringify({comment:text})});
    if(r.status===401) throw new Error('API key required or invalid.');
    if(r.status===429) throw new Error('Rate limit hit. Retry in '+(r.headers.get('Retry-After')||'a few')+' s.');
    if(!r.ok) throw new Error('Request failed ('+r.status+').');
    const d=await r.json();
    $('lab').textContent='Prediction: '+d.label_name.replace(/_/g,' ');
    const bars=$('bars'); bars.textContent='';
    for(const [name,p] of Object.entries(d.probabilities)){
      const row=document.createElement('div'); row.className='bar';
      const n=document.createElement('span'); n.className='n'; n.textContent=name.replace(/_/g,' ');
      const tr=document.createElement('div'); tr.className='track';
      const f=document.createElement('div'); f.className='fill'; f.style.width=(p*100).toFixed(1)+'%'; f.style.background=COLORS[name]||'var(--accent)';
      const v=document.createElement('span'); v.textContent=(p*100).toFixed(1)+'%';
      tr.appendChild(f); row.append(n,tr,v); bars.appendChild(row);
    }
    $('result').style.display='block';
  }catch(e){$('err').textContent=e.message;$('result').style.display='none';}
  finally{$('go').disabled=false;}
};
</script></body></html>"""
