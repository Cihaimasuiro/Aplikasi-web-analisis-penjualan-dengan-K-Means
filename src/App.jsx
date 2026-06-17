import React, { useState, useMemo, useCallback } from 'react';
import Papa from 'papaparse';
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis,
} from 'recharts';

const C = {
  bg: '#11151C', surface: '#1A2029', surfaceAlt: '#222B38', border: '#2E3848',
  amber: '#E3A23C', teal: '#4FD1C5', rose: '#E8765C', violet: '#9D8DF1',
  green: '#7FD88F', blue: '#6B9BD1', text: '#F2EFE9', muted: '#8C96A8',
};
const CLUSTER_COLORS = [C.amber, C.teal, C.rose, C.violet, C.green, C.blue, '#D1A3D8', '#F2C572'];

const fmtIDR = (v) => new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(v || 0);
const fmtNum = (v, d = 0) => new Intl.NumberFormat('id-ID', { maximumFractionDigits: d }).format(v || 0);

function parseDate(val) {
  if (val instanceof Date) return isNaN(val.getTime()) ? null : val;
  if (val == null || val === '') return null;
  const s = String(val).trim();
  let d = new Date(s);
  if (!isNaN(d.getTime())) return d;
  const m = s.match(/^(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})/);
  if (m) {
    let [, a, b, c] = m;
    if (c.length === 2) c = '20' + c;
    d = new Date(`${c}-${b.padStart(2,'0')}-${a.padStart(2,'0')}`);
    if (!isNaN(d.getTime())) return d;
  }
  return null;
}

function euclideanDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) { const diff = a[i] - b[i]; sum += diff * diff; }
  return Math.sqrt(sum);
}

function initCentroidsPlusPlus(data, k) {
  const centroids = [data[Math.floor(Math.random() * data.length)]];
  while (centroids.length < k) {
    const dists = data.map(p => { let min = Infinity; for (const c of centroids) { const d = euclideanDist(p,c); if (d<min) min=d; } return min*min; });
    const sum = dists.reduce((a,b) => a+b, 0);
    if (sum === 0) { centroids.push(data[Math.floor(Math.random()*data.length)]); continue; }
    let r = Math.random()*sum, acc=0, chosen=data[0];
    for (let i=0; i<data.length; i++) { acc+=dists[i]; if (acc>=r) { chosen=data[i]; break; } }
    centroids.push(chosen);
  }
  return centroids.map(c => [...c]);
}

function runKMeans(data, k, maxIter=50, restarts=4) {
  if (!data.length || k <= 0) return { assignments:[], centroids:[], wcss:0 };
  const effK = Math.max(1, Math.min(k, data.length));
  let best = null;
  for (let r=0; r<restarts; r++) {
    let centroids = initCentroidsPlusPlus(data, effK);
    let assignments = new Array(data.length).fill(0);
    for (let iter=0; iter<maxIter; iter++) {
      let changed = false;
      for (let i=0; i<data.length; i++) {
        let bestDist=Infinity, bestC=0;
        for (let c=0; c<effK; c++) { const d=euclideanDist(data[i],centroids[c]); if (d<bestDist){bestDist=d;bestC=c;} }
        if (assignments[i]!==bestC) { assignments[i]=bestC; changed=true; }
      }
      if (!changed && iter > 0) {
        // Hitung ulang pusat kelompok satu kali sebelum berhenti.
        const dim=data[0].length;
        const sums=Array.from({length:effK},()=>new Array(dim).fill(0));
        const counts=new Array(effK).fill(0);
        for (let i=0; i<data.length; i++) { counts[assignments[i]]++; for (let j=0; j<dim; j++) sums[assignments[i]][j]+=data[i][j]; }
        for (let c=0; c<effK; c++) { if (counts[c]>0) centroids[c]=sums[c].map(s=>s/counts[c]); }
        break;
      }
      const dim=data[0].length;
      const sums=Array.from({length:effK},()=>new Array(dim).fill(0));
      const counts=new Array(effK).fill(0);
      for (let i=0; i<data.length; i++) { counts[assignments[i]]++; for (let j=0; j<dim; j++) sums[assignments[i]][j]+=data[i][j]; }
      for (let c=0; c<effK; c++) { if (counts[c]>0) centroids[c]=sums[c].map(s=>s/counts[c]); }
    }
    let wcss=0;
    for (let i=0; i<data.length; i++) {
      if (centroids[assignments[i]]) {
        wcss+=euclideanDist(data[i],centroids[assignments[i]])**2;
      }
    }
    if (!best||wcss<best.wcss) best={assignments,centroids,wcss};
  }
  return best;
}

function calculateSilhouetteScore(data, assignments, k) {
  if (k <= 1 || data.length < k) return 0;
  let totalScore = 0;
  const pointScores = new Array(data.length).fill(0);

  for (let i = 0; i < data.length; i++) {
    const myCluster = assignments[i];
    let a = 0, aCount = 0;
    const b = new Array(k).fill(0);
    const bCounts = new Array(k).fill(0);

    for (let j = 0; j < data.length; j++) {
      if (i === j) continue;
      const dist = euclideanDist(data[i], data[j]);
      const otherCluster = assignments[j];
      if (myCluster === otherCluster) {
        a += dist;
        aCount++;
      } else {
        b[otherCluster] += dist;
        bCounts[otherCluster]++;
      }
    }

    const avgA = aCount > 0 ? a / aCount : 0;
    
    let minAvgB = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === myCluster) continue;
      const avgB = bCounts[c] > 0 ? b[c] / bCounts[c] : 0;
      if (bCounts[c] > 0 && avgB < minAvgB) {
        minAvgB = avgB;
      }
    }
    if (minAvgB === Infinity) minAvgB = 0;

    const denominator = Math.max(avgA, minAvgB);
    pointScores[i] = denominator === 0 ? 0 : (minAvgB - avgA) / denominator;
  }

  totalScore = pointScores.reduce((sum, score) => sum + score, 0);
  return data.length > 0 ? totalScore / data.length : 0;
}


function getSegmentLabels(k) {
  const base = ['Pelanggan Utama','Pelanggan Setia','Pelanggan Reguler','Pelanggan Potensial','Pelanggan Baru'];
  if (k===1) return ['Seluruh Pelanggan'];
  if (k-1<=base.length) return [...base.slice(0,k-1),'Pelanggan Berisiko (Berisiko)'];
  const extra=[];
  for (let i=0; i<k-1-base.length; i++) extra.push(`Segmen Tambahan ${i+1}`);
  return [...base,...extra,'Pelanggan Berisiko (Berisiko)'];
}

// Klasifikasi tingkat (Tinggi/Sedang/Rendah) relatif terhadap rata-rata keseluruhan pelanggan.
// inverse=true dipakai untuk kebaruan transaksi, karena nilai hari yang lebih kecil berarti lebih aktif.
function classifyLevel(value, overallAvg, inverse = false) {
  if (!overallAvg) return 'Sedang';
  const ratio = value / overallAvg;
  if (inverse) {
    if (ratio <= 0.7) return 'Tinggi';
    if (ratio >= 1.3) return 'Rendah';
    return 'Sedang';
  }
  if (ratio >= 1.2) return 'Tinggi';
  if (ratio <= 0.8) return 'Rendah';
  return 'Sedang';
}

function buildNarrative(s, levelR, levelF, levelM, pctCustomers, pctRevenue) {
  const recencyDesc = levelR === 'Tinggi'
    ? `masih aktif bertransaksi baru-baru ini (rata-rata ${fmtNum(s.avgR, 1)} hari sejak transaksi terakhir)`
    : levelR === 'Rendah'
    ? `sudah cukup lama tidak bertransaksi (rata-rata ${fmtNum(s.avgR, 1)} hari sejak transaksi terakhir)`
    : `tergolong cukup aktif (rata-rata ${fmtNum(s.avgR, 1)} hari sejak transaksi terakhir)`;
  const freqDesc = levelF === 'Tinggi'
    ? `frekuensi transaksi yang tinggi (rata-rata ${fmtNum(s.avgF, 1)} kali)`
    : levelF === 'Rendah'
    ? `frekuensi transaksi yang rendah (rata-rata ${fmtNum(s.avgF, 1)} kali)`
    : `frekuensi transaksi yang sedang (rata-rata ${fmtNum(s.avgF, 1)} kali)`;
  const monDesc = levelM === 'Tinggi'
    ? `nilai belanja yang tinggi (rata-rata ${fmtIDR(s.avgM)} per pelanggan)`
    : levelM === 'Rendah'
    ? `nilai belanja yang rendah (rata-rata ${fmtIDR(s.avgM)} per pelanggan)`
    : `nilai belanja yang sedang (rata-rata ${fmtIDR(s.avgM)} per pelanggan)`;
  return `Segmen ini mencakup ${fmtNum(s.count)} pelanggan (${fmtNum(pctCustomers, 1)}% dari total pelanggan) dan berkontribusi sekitar ${fmtNum(pctRevenue, 1)}% dari estimasi total pendapatan seluruh pelanggan. Secara karakteristik, pelanggan pada segmen ini ${recencyDesc}, memiliki ${freqDesc}, serta menunjukkan ${monDesc}.`;
}

function buildRecommendations(levelR, levelF, levelM) {
  if (levelR === 'Rendah') {
    return [
      'Lakukan kampanye pengaktifan kembali melalui email/SMS/WhatsApp dengan penawaran diskon khusus untuk mendorong transaksi ulang.',
      'Kirim survei singkat untuk memahami alasan menurunnya aktivitas dan perbaiki titik gesekan dalam pengalaman belanja.',
      'Tawarkan promo khusus pelanggan lama dengan masa berlaku terbatas agar pelanggan terdorong kembali bertransaksi.',
    ];
  }
  if (levelF === 'Tinggi' && levelM === 'Tinggi') {
    return [
      'Berikan program loyalitas eksklusif seperti tingkatan VIP, akses awal produk baru, atau hadiah khusus.',
      'Libatkan sebagai promotor merek melalui program rujukan dengan insentif yang menarik.',
      'Tingkatkan komunikasi yang dipersonalisasi, misalnya rekomendasi produk yang relevan, untuk menjaga keterikatan pelanggan.',
    ];
  }
  if (levelM === 'Tinggi' && levelF !== 'Tinggi') {
    return [
      'Dorong frekuensi belanja dengan penawaran paket atau diskon untuk pembelian berikutnya dalam jangka waktu tertentu.',
      'Sediakan layanan personal atau akun khusus karena nilai transaksi mereka besar.',
      'Kirim pengingat pembelian ulang atau rekomendasi produk pelengkap berdasarkan riwayat pembelian sebelumnya.',
    ];
  }
  if (levelF === 'Tinggi' && levelM !== 'Tinggi') {
    return [
      'Terapkan strategi penjualan produk pelengkap dan peningkatan paket belanja untuk menaikkan nilai pesanan rata-rata.',
      'Tawarkan paket produk atau diskon bertingkat agar nilai belanja per transaksi meningkat.',
      'Perkenalkan kategori atau produk premium yang relevan dengan kebiasaan belanja mereka yang sering.',
    ];
  }
  if (levelF === 'Rendah' && levelM === 'Rendah' && levelR === 'Tinggi') {
    return [
      'Berikan panduan awal atau edukasi produk untuk meningkatkan keterlibatan pelanggan yang relatif baru ini.',
      'Tawarkan insentif pembelian kedua (diskon khusus dalam 30 hari pertama) untuk mendorong pembelian berulang.',
      'Kumpulkan umpan balik awal untuk memahami ekspektasi dan menyesuaikan pengalaman pelanggan ke depannya.',
    ];
  }
  return [
    'Pertahankan keterlibatan dengan program loyalitas berbasis poin untuk mendorong pelanggan naik ke segmen yang lebih bernilai.',
    'Lakukan kampanye edukasi produk secara bertahap untuk meningkatkan frekuensi dan nilai transaksi.',
    'Pantau perubahan perilaku segmen ini secara berkala karena posisinya di tengah berpotensi bergerak ke segmen lain.',
  ];
}

const FIELD_HINTS = {
  customerId: ['idpelanggan','id_pelanggan','pelanggan'],
  date: ['tanggal','tanggaltransaksi','tgl'],
  invoiceId: ['notransaksi','no_transaksi','nofaktur','no_faktur'],
  product: ['namaproduk','nama_produk','produk'],
  quantity: ['jumlah','jumlahbarang','jml'],
  price: ['harga','hargasatuan','harga_satuan'],
};
function autoDetectMapping(headers) {
  const norm = s => String(s).toLowerCase().replace(/[\s\-_.]/g,'');
  const mapping = {};
  for (const field of Object.keys(FIELD_HINTS)) {
    mapping[field] = headers.find(h => FIELD_HINTS[field].includes(norm(h))) || '';
  }
  return mapping;
}

const tooltipStyle = { background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 6, fontFamily: '"IBM Plex Mono",monospace', fontSize: 12, color: C.text };

function Ticket({ children, style={} }) {
  return <div style={{ position:'relative', background:C.surface, border:`1px solid ${C.border}`, borderRadius:6, padding:'18px 20px', ...style }}>{children}</div>;
}

function Kpi({ label, value, accent }) {
  return (
    <Ticket>
      <div style={{ fontSize:11, letterSpacing:'0.12em', textTransform:'uppercase', color:C.muted, fontFamily:'Inter,sans-serif' }}>{label}</div>
      <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:26, fontWeight:600, color:accent||C.text, marginTop:6 }}>{value}</div>
    </Ticket>
  );
}

function SectionTitle({ eyebrow, title, desc }) {
  return (
    <div style={{ marginBottom:20 }}>
      {eyebrow && <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, color:C.amber, letterSpacing:'0.15em', textTransform:'uppercase', marginBottom:6 }}>{eyebrow}</div>}
      <h2 style={{ fontFamily:'"Fraunces",serif', fontSize:26, fontWeight:600, color:C.text, margin:0 }}>{title}</h2>
      {desc && <p style={{ color:C.muted, fontSize:14, marginTop:8, maxWidth:720, lineHeight:1.6 }}>{desc}</p>}
    </div>
  );
}

const MAPPING_FIELDS = [
  { key:'customerId', label:'ID Pelanggan', required:true, desc:'Identitas unik pelanggan' },
  { key:'date', label:'Tanggal Transaksi', required:true, desc:'Digunakan untuk menghitung kebaruan transaksi' },
  { key:'quantity', label:'Jumlah Barang', required:true, desc:'Jumlah unit terjual' },
  { key:'price', label:'Harga Satuan', required:true, desc:'Harga per unit' },
  { key:'invoiceId', label:'ID Transaksi / Faktur', required:false, desc:'Untuk menghitung frekuensi transaksi (opsional)' },
  { key:'product', label:'Nama Produk', required:false, desc:'Untuk ringkasan produk terlaris (opsional)' },
];

const AXIS_OPTIONS = [
  { key:'recency', label:'Kebaruan transaksi (hari)' },
  { key:'frequency', label:'Frekuensi (kali transaksi)' },
  { key:'monetary', label:'Nilai belanja (Rp)' },
];

export default function App() {
  const [stage, setStage] = useState('upload');
  const [fileName, setFileName] = useState('');
  const [headers, setHeaders] = useState([]);
  const [rawData, setRawData] = useState([]);
  const [mapping, setMapping] = useState({});
  const [parseError, setParseError] = useState('');
  const [activeTab, setActiveTab] = useState('ringkasan');
  const [kValue, setKValue] = useState(3);
  const [evaluationData, setEvaluationData] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [clusterResult, setClusterResult] = useState(null);
  const [clusterLoading, setClusterLoading] = useState(false);
  const [axisX, setAxisX] = useState('recency');
  const [axisY, setAxisY] = useState('monetary');
  const [showRfmTable, setShowRfmTable] = useState(false);
  const [rfmCurrentPage, setRfmCurrentPage] = useState(1);
  const RFM_PAGE_SIZE = 10;

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setParseError('');
    setFileName(file.name);
    Papa.parse(file, {
      header:true, dynamicTyping:true, skipEmptyLines:true,
      complete: (results) => {
        const hdrs = results.meta.fields || [];
        if (!hdrs.length || !results.data.length) { setParseError('File CSV tidak terbaca atau kosong.'); return; }
        setHeaders(hdrs); setRawData(results.data); setMapping(autoDetectMapping(hdrs)); setStage('mapping');
      },
      error: () => setParseError('Gagal membaca file. Pastikan format .csv'),
    });
  };

  const resetAll = () => { setStage('upload'); setFileName(''); setHeaders([]); setRawData([]); setMapping({}); setEvaluationData(null); setClusterResult(null); setActiveTab('ringkasan'); };

  const transactions = useMemo(() => {
    if (stage!=='app') return [];
    const { customerId, date, quantity, price, invoiceId, product } = mapping;
    const out = [];
    for (const row of rawData) {
      const cust = row[customerId];
      const qty = Number(row[quantity]);
      const prc = Number(row[price]);
      const d = parseDate(row[date]);
      if (cust==null||cust===''||!d||!isFinite(qty)||!isFinite(prc)||qty<=0||prc<=0) continue;
      out.push({ customerId:String(cust), date:d, amount:qty*prc, invoiceId:invoiceId?row[invoiceId]:null, product:product?String(row[product]):null });
    }
    return out;
  }, [rawData, mapping, stage]);

  const customers = useMemo(() => {
    if (!transactions.length) return [];
    let maxDate = transactions[0].date;
    for (const t of transactions) if (t.date>maxDate) maxDate=t.date;
    const refDate = new Date(maxDate.getTime()); refDate.setDate(refDate.getDate()+1);
    const map = new Map();
    for (const t of transactions) {
      if (!map.has(t.customerId)) map.set(t.customerId,{customerId:t.customerId,lastDate:t.date,invoices:new Set(),txCount:0,monetary:0});
      const c=map.get(t.customerId);
      if (t.date>c.lastDate) c.lastDate=t.date;
      if (t.invoiceId!=null&&t.invoiceId!=='') c.invoices.add(t.invoiceId);
      c.txCount++; c.monetary+=t.amount;
    }
    const hasInvoice=!!mapping.invoiceId;
    return Array.from(map.values()).map(c=>({ customerId:c.customerId, recency:Math.max(0,Math.round((refDate-c.lastDate)/86400000)), frequency:hasInvoice?Math.max(c.invoices.size,1):c.txCount, monetary:Math.round(c.monetary) }));
  }, [transactions, mapping.invoiceId]);

  const normalizedFeatures = useMemo(() => {
    if (!customers.length) return [];
    const cols=['recency','frequency','monetary'];
    const mins={},maxs={};
    cols.forEach(col=>{ const vals=customers.map(c=>c[col]); mins[col]=Math.min(...vals); maxs[col]=Math.max(...vals); });
    return customers.map(c=>cols.map(col=>{ const range=maxs[col]-mins[col]; return range===0?0:(c[col]-mins[col])/range; }));
  }, [customers]);

  const overview = useMemo(() => {
    if (!transactions.length) return null;
    const totalRevenue=transactions.reduce((s,t)=>s+t.amount,0);
    const invoiceSet=new Set(), customerSet=new Set(), productMap=new Map(), monthMap=new Map();
    transactions.forEach(t=>{
      if (t.invoiceId!=null&&t.invoiceId!=='') invoiceSet.add(t.invoiceId);
      customerSet.add(t.customerId);
      const pname=t.product||'Tidak diketahui';
      productMap.set(pname,(productMap.get(pname)||0)+t.amount);
      const ym=`${t.date.getFullYear()}-${String(t.date.getMonth()+1).padStart(2,'0')}`;
      monthMap.set(ym,(monthMap.get(ym)||0)+t.amount);
    });
    return {
      totalRevenue, totalOrders:invoiceSet.size||transactions.length,
      totalCustomers:customerSet.size, totalProducts:productMap.size,
      avgOrder:totalRevenue/(invoiceSet.size||transactions.length),
      topProducts:Array.from(productMap.entries()).map(([name,value])=>({name:name.length>18?name.slice(0,18)+'...':name,value})).sort((a,b)=>b.value-a.value).slice(0,8),
      monthly:Array.from(monthMap.entries()).map(([month,value])=>({month,value})).sort((a,b)=>a.month.localeCompare(b.month)),
    };
  }, [transactions]);

  const runEvaluation = useCallback(() => {
    if (!normalizedFeatures.length) return;
    setEvaluationLoading(true);
    setTimeout(() => {
      const maxK = Math.min(8, normalizedFeatures.length - 1);
      const results = [];
      for (let k = 2; k <= maxK; k++) {
        const kmeansResult = runKMeans(normalizedFeatures, k, 50, 3);
        const wcss = Math.round(kmeansResult.wcss * 100) / 100;
        const silhouette = calculateSilhouetteScore(normalizedFeatures, kmeansResult.assignments, k);
        results.push({ k, wcss, silhouette: Math.round(silhouette * 1000) / 1000 });
      }
      setEvaluationData(results);
      setEvaluationLoading(false);
    }, 30);
  }, [normalizedFeatures]);

  const runClustering = useCallback(() => {
    if (!normalizedFeatures.length) return;
    setClusterLoading(true);
    setTimeout(()=>{ setClusterResult(runKMeans(normalizedFeatures,kValue,80,6)); setClusterLoading(false); },30);
  }, [normalizedFeatures, kValue]);

  const clusterSummary = useMemo(() => {
    if (!clusterResult||!clusterResult.assignments.length) return null;
    const k=Math.min(kValue,normalizedFeatures.length);
    const groups=Array.from({length:k},()=>({count:0,sumR:0,sumF:0,sumM:0}));
    customers.forEach((cust,i)=>{ const c=clusterResult.assignments[i]; groups[c].count++; groups[c].sumR+=cust.recency; groups[c].sumF+=cust.frequency; groups[c].sumM+=cust.monetary; });
    const stats=groups.map((g,idx)=>({ kelompok:idx, count:g.count, avgR:g.count?g.sumR/g.count:0, avgF:g.count?g.sumF/g.count:0, avgM:g.count?g.sumM/g.count:0 }));
    const maxR=Math.max(...stats.map(s=>s.avgR),1), maxF=Math.max(...stats.map(s=>s.avgF),1), maxM=Math.max(...stats.map(s=>s.avgM),1);
    const scored=stats.map(s=>({...s,score:s.avgF/maxF+s.avgM/maxM-s.avgR/maxR}));
    const sorted=[...scored].sort((a,b)=>b.score-a.score);
    const labels=getSegmentLabels(k);
    const labelMap={}; sorted.forEach((s,rank)=>(labelMap[s.kelompok]=labels[rank]));
    return scored.map(s=>({...s,label:labelMap[s.kelompok]})).sort((a,b)=>a.kelompok-b.kelompok);
  }, [clusterResult,customers,kValue,normalizedFeatures.length]);

  const scatterData = useMemo(() => { if (!clusterResult||!clusterResult.assignments.length) return []; return customers.map((c,i)=>({...c,kelompok:clusterResult.assignments[i]})); }, [clusterResult,customers]);
  const pieData = useMemo(() => { if (!clusterSummary) return []; return clusterSummary.map(s=>({name:s.label,value:s.count,kelompok:s.kelompok})); }, [clusterSummary]);

  const overallStats = useMemo(() => {
    if (!customers.length) return null;
    const n = customers.length;
    return {
      avgR: customers.reduce((s,c)=>s+c.recency,0)/n,
      avgF: customers.reduce((s,c)=>s+c.frequency,0)/n,
      avgM: customers.reduce((s,c)=>s+c.monetary,0)/n,
      totalM: customers.reduce((s,c)=>s+c.monetary,0),
    };
  }, [customers]);

  const clusterInterpretations = useMemo(() => {
    if (!clusterSummary || !overallStats || !customers.length) return null;
    return clusterSummary.map(s => {
      const levelR = classifyLevel(s.avgR, overallStats.avgR, true);
      const levelF = classifyLevel(s.avgF, overallStats.avgF, false);
      const levelM = classifyLevel(s.avgM, overallStats.avgM, false);
      const pctCustomers = (s.count / customers.length) * 100;
      const segmentRevenue = s.avgM * s.count;
      const pctRevenue = overallStats.totalM > 0 ? (segmentRevenue / overallStats.totalM) * 100 : 0;
      return {
        ...s, levelR, levelF, levelM, pctCustomers, pctRevenue, segmentRevenue,
        narrative: buildNarrative(s, levelR, levelF, levelM, pctCustomers, pctRevenue),
        recommendations: buildRecommendations(levelR, levelF, levelM),
      };
    });
  }, [clusterSummary, overallStats, customers.length]);

  const executiveSummary = useMemo(() => {
    if (!clusterInterpretations) return null;
    const top = [...clusterInterpretations].sort((a,b)=>b.pctRevenue-a.pctRevenue)[0];
    const atRisk = [...clusterInterpretations].filter(s=>s.levelR==='Rendah').sort((a,b)=>b.count-a.count)[0];
    return { top, atRisk };
  }, [clusterInterpretations]);

  const downloadResults = () => {
    if (!clusterResult||!clusterSummary) return;
    const labelMap={}; clusterSummary.forEach(s=>(labelMap[s.kelompok]=s.label));
    const rows=customers.map((c,i)=>({ID_Pelanggan:c.customerId,Kebaruan_Transaksi_Hari:c.recency,Frekuensi_Transaksi:c.frequency,Nilai_Belanja:c.monetary,Kelompok:clusterResult.assignments[i],Segmen:labelMap[clusterResult.assignments[i]]}));
    const csv=Papa.unparse(rows);
    const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download='hasil_segmentasi_kmeans.csv'; a.click(); URL.revokeObjectURL(url);
  };

  const downloadSummaryReport = () => {
    if (!clusterInterpretations) return;
    const rows = clusterInterpretations.map(s => ({
      Kelompok: s.kelompok,
      Segmen: s.label,
      Jumlah_Pelanggan: s.count,
      Persentase_Pelanggan: `${fmtNum(s.pctCustomers, 1)}%`,
      Rata_Rata_Kebaruan_Transaksi_Hari: fmtNum(s.avgR, 1),
      Tingkat_Kebaruan_Transaksi: s.levelR,
      Rata_Rata_Frekuensi: fmtNum(s.avgF, 1),
      Tingkat_Frekuensi: s.levelF,
      Rata_Rata_Nilai_Belanja: s.avgM,
      Tingkat_Nilai_Belanja: s.levelM,
      Persentase_Kontribusi_Pendapatan: `${fmtNum(s.pctRevenue, 1)}%`,
      Deskripsi: s.narrative,
      Rekomendasi_Strategi: s.recommendations.join(' | '),
    }));
    const csv = Papa.unparse(rows);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'ringkasan_interpretasi_segmen.csv'; a.click(); URL.revokeObjectURL(url);
  };

  if (stage==='upload') return (
    <div style={{ minHeight:'100vh', background:C.bg, color:C.text, display:'flex', alignItems:'center', justifyContent:'center', padding:24, fontFamily:'Inter,sans-serif' }}>
      <div style={{ maxWidth:620, width:'100%' }}>
        <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, color:C.amber, letterSpacing:'0.2em', textTransform:'uppercase', marginBottom:14 }}>Sistem Analisis Data Penjualan Ritel</div>
        <h1 style={{ fontFamily:'"Fraunces",serif', fontSize:44, fontWeight:600, lineHeight:1.15, margin:0, marginBottom:16 }}>Segmentasi pelanggan berbasis <span style={{color:C.amber}}>K-Means</span></h1>
        <p style={{ color:C.muted, fontSize:15, lineHeight:1.7, marginBottom:32 }}>Unggah data transaksi penjualan (.csv) untuk melihat ringkasan penjualan dan menjalankan segmentasi pelanggan berdasarkan analisis RFM menggunakan algoritma K-Means.</p>
        <Ticket style={{ padding:28, borderStyle:'dashed' }}>
          <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:13, color:C.muted, marginBottom:14 }}>01 - UNGGAH FILE CSV</div>
          <input type="file" accept=".csv" onChange={handleFile} style={{ color:C.muted, fontSize:13 }} />
          {parseError && <div style={{ color:C.rose, fontSize:13, marginTop:14 }}>{parseError}</div>}
          <div style={{ marginTop:18, paddingTop:18, borderTop:`1px solid ${C.border}`, fontSize:12, color:C.muted, lineHeight:1.7 }}>Kolom yang didukung: ID Pelanggan, Tanggal, Jumlah Barang, Harga Satuan, ID Transaksi/Faktur (opsional), Nama Produk (opsional).</div>
        </Ticket>
      </div>
    </div>
  );

  if (stage==='mapping') {
    const requiredOk = MAPPING_FIELDS.filter(f=>f.required).every(f=>mapping[f.key]);
    const preview = rawData.slice(0,5);
    return (
      <div style={{ minHeight:'100vh', background:C.bg, color:C.text, padding:'40px 48px', maxWidth:1000, margin:'0 auto', fontFamily:'Inter,sans-serif' }}>
        <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, color:C.amber, letterSpacing:'0.2em', textTransform:'uppercase', marginBottom:10 }}>02 - PEMETAAN KOLOM</div>
        <h1 style={{ fontFamily:'"Fraunces",serif', fontSize:32, fontWeight:600, margin:0, marginBottom:8 }}>{fileName}</h1>
        <p style={{ color:C.muted, fontSize:14, marginBottom:28 }}>{rawData.length.toLocaleString('id-ID')} baris terdeteksi.</p>
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:32 }}>
          {MAPPING_FIELDS.map(f=>(
            <Ticket key={f.key}>
              <label style={{ display:'block', fontSize:13, fontWeight:600, marginBottom:4 }}>{f.label} {f.required&&<span style={{color:C.amber}}>*</span>}</label>
              <div style={{ fontSize:12, color:C.muted, marginBottom:10 }}>{f.desc}</div>
              <select value={mapping[f.key]||''} onChange={e=>setMapping(m=>({...m,[f.key]:e.target.value}))} style={{ width:'100%', background:C.surfaceAlt, color:C.text, border:`1px solid ${C.border}`, borderRadius:5, padding:'8px 10px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13 }}>
                <option value="">{f.required?'- pilih kolom -':'(tidak digunakan)'}</option>
                {headers.map(h=><option key={h} value={h}>{h}</option>)}
              </select>
            </Ticket>
          ))}
        </div>
        <Ticket style={{ marginBottom:32, overflowX:'auto' }}>
          <div style={{ fontSize:12, color:C.muted, marginBottom:10, fontFamily:'"IBM Plex Mono",monospace' }}>PRATINJAU DATA (5 BARIS)</div>
          <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
            <thead><tr>{headers.slice(0,6).map(h=><th key={h} style={{ textAlign:'left', padding:'10px 14px', color:C.muted, borderBottom:`1px solid ${C.border}` }}>{h}</th>)}</tr></thead>
            <tbody>{preview.map((row,i)=><tr key={i}>{headers.slice(0,6).map(h=><td key={h} style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{String(row[h]??'')}</td>)}</tr>)}</tbody>
          </table>
        </Ticket>
        <div style={{ display:'flex', gap:12 }}>
          <button onClick={resetAll} style={{ background:'transparent', color:C.text, border:`1px solid ${C.border}`, borderRadius:6, padding:'10px 20px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13, cursor:'pointer' }}>Kembali ke Unggah File</button>
          <button disabled={!requiredOk} onClick={()=>setStage('app')} style={{ background:C.amber, color:'#1A1306', border:'none', borderRadius:6, padding:'11px 22px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13, fontWeight:600, cursor:requiredOk?'pointer':'not-allowed', opacity:requiredOk?1:0.4 }}>Proses dan Lanjutkan</button>
        </div>
        {!requiredOk && <div style={{ color:C.rose, fontSize:12, marginTop:10 }}>Lengkapi kolom wajib (*) sebelum melanjutkan.</div>}
      </div>
    );
  }

  return (
    <div style={{ display:'flex', minHeight:'100vh', background:C.bg, color:C.text, fontFamily:'Inter,sans-serif' }}>
      {/* SIDEBAR */}
      <aside style={{ width:240, background:C.surface, borderRight:`1px solid ${C.border}`, padding:'28px 18px', display:'flex', flexDirection:'column', flexShrink:0 }}>
        <div style={{ padding:'0 6px', marginBottom:28 }}>
          <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:11, color:C.amber, letterSpacing:'0.2em', textTransform:'uppercase' }}>Analisis Ritel</div>
          <div style={{ fontFamily:'"Fraunces",serif', fontSize:19, fontWeight:600, marginTop:4 }}>Dasbor K-Means</div>
        </div>
        <nav style={{ display:'flex', flexDirection:'column', gap:4, marginBottom:28 }}>
          {[['ringkasan','Ringkasan Penjualan'],['segmentasi','Segmentasi Pelanggan']].map(([tab,label])=>(
            <button key={tab} onClick={()=>setActiveTab(tab)} style={{ display:'flex', alignItems:'center', gap:10, width:'100%', textAlign:'left', padding:'11px 14px', borderRadius:6, background:activeTab===tab?C.surfaceAlt:'transparent', border:'none', cursor:'pointer', color:activeTab===tab?C.amber:C.muted, fontFamily:'Inter,sans-serif', fontSize:14, fontWeight:500 }}>{label}</button>
          ))}
        </nav>
        <div style={{ marginTop:'auto' }}>
          <Ticket style={{ padding:14, marginBottom:12 }}>
            <div style={{ fontSize:11, color:C.muted, marginBottom:4, textTransform:'uppercase', letterSpacing:'0.08em' }}>File aktif</div>
            <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, wordBreak:'break-all', marginBottom:6 }}>{fileName}</div>
            <div style={{ fontSize:11, color:C.muted }}>{transactions.length.toLocaleString('id-ID')} transaksi valid</div>
          </Ticket>
          <button onClick={resetAll} style={{ width:'100%', background:'transparent', color:C.text, border:`1px solid ${C.border}`, borderRadius:6, padding:'10px 20px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13, cursor:'pointer' }}>Unggah File Lain</button>
        </div>
      </aside>

      {/* MAIN */}
      <main style={{ flex:1, padding:'32px 40px', minWidth:0, overflowY:'auto' }}>
        {activeTab==='ringkasan' && (
          overview ? (
            <div>
              <SectionTitle eyebrow="Dasbor" title="Ringkasan Penjualan" desc="Gambaran umum performa penjualan berdasarkan data transaksi yang diunggah." />
              <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:18, marginBottom:32 }}>
                <Kpi label="Total Pendapatan" value={fmtIDR(overview.totalRevenue)} accent={C.amber} />
                <Kpi label="Total Transaksi" value={fmtNum(overview.totalOrders)} />
                <Kpi label="Total Pelanggan" value={fmtNum(overview.totalCustomers)} />
                <Kpi label="Rata-rata / Transaksi" value={fmtIDR(overview.avgOrder)} />
              </div>
              <div style={{ display:'grid', gridTemplateColumns:'1.4fr 1fr', gap:18 }}>
                <Ticket style={{ padding:22 }}>
                  <div style={{ fontSize:12, color:C.muted, marginBottom:16, fontFamily:'"IBM Plex Mono",monospace', letterSpacing:'0.08em' }}>TREN PENDAPATAN BULANAN</div>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={overview.monthly}>
                      <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
                      <XAxis dataKey="month" stroke={C.muted} fontSize={11} />
                      <YAxis stroke={C.muted} fontSize={11} tickFormatter={v=>v>=1e6?`${(v/1e6).toFixed(0)}jt`:v} />
                      <Tooltip contentStyle={tooltipStyle} formatter={v=>fmtIDR(v)} />
                      <Line type="monotone" dataKey="value" stroke={C.amber} strokeWidth={2.5} dot={{r:3,fill:C.amber}} name="Pendapatan" />
                    </LineChart>
                  </ResponsiveContainer>
                </Ticket>
                <Ticket style={{ padding:22 }}>
                  <div style={{ fontSize:12, color:C.muted, marginBottom:16, fontFamily:'"IBM Plex Mono",monospace', letterSpacing:'0.08em' }}>PRODUK TERLARIS (BERDASARKAN PENDAPATAN)</div>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={overview.topProducts} layout="vertical" margin={{left:10}}>
                      <CartesianGrid stroke={C.border} strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" stroke={C.muted} fontSize={10} tickFormatter={v=>v>=1e6?`${(v/1e6).toFixed(0)}jt`:v} />
                      <YAxis type="category" dataKey="name" stroke={C.muted} fontSize={10} width={100} />
                      <Tooltip contentStyle={tooltipStyle} formatter={v=>fmtIDR(v)} />
                      <Bar dataKey="value" fill={C.teal} radius={[0,3,3,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Ticket>
              </div>
            </div>
          ) : <Ticket style={{padding:24}}>Tidak ada data transaksi valid. Periksa kembali pemetaan kolom.</Ticket>
        )}

        {activeTab==='segmentasi' && (
          customers.length===0 ? <Ticket style={{padding:24}}>Tidak ada data pelanggan yang dapat dianalisis.</Ticket> : (
            <div>
              <SectionTitle eyebrow="Pengelompokan K-Means" title="Segmentasi Pelanggan (RFM)" desc={`${customers.length.toLocaleString('id-ID')} pelanggan diukur berdasarkan kebaruan transaksi, frekuensi transaksi, dan nilai belanja, lalu dinormalisasi (min-max) dan dikelompokkan dengan K-Means.`} />

              <Ticket style={{ padding:22, marginBottom:24 }}>
                <button onClick={() => setShowRfmTable(s => !s)} style={{ width: '100%', background: 'none', border: 'none', color: C.text, padding: 0, textAlign: 'left', cursor: 'pointer' }}>
                  <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                    <div style={{ fontFamily:'"Fraunces",serif', fontSize:19, fontWeight:600 }}>Analisis RFM</div>
                    <div style={{ transform: showRfmTable ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>v</div>
                  </div>
                  <div style={{ fontSize:13, color:C.muted, marginTop:4 }}>Tabel rincian kebaruan transaksi, frekuensi transaksi, dan nilai belanja untuk setiap pelanggan.</div>
                </button>
                {showRfmTable && (
                  <div style={{ marginTop: 20, paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
                    <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
                      <thead><tr>{['ID Pelanggan','Kebaruan transaksi (hari)','Frekuensi (kali)','Nilai belanja (Rp)'].map(h=><th key={h} style={{ textAlign:'left', padding:'10px 14px', color:C.muted, borderBottom:`1px solid ${C.border}`, fontSize:11, textTransform:'uppercase', letterSpacing:'0.08em' }}>{h}</th>)}</tr></thead>
                      <tbody>
                        {customers.slice((rfmCurrentPage - 1) * RFM_PAGE_SIZE, rfmCurrentPage * RFM_PAGE_SIZE).map(c=>(
                          <tr key={c.customerId}>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{c.customerId}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{fmtNum(c.recency)}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{fmtNum(c.frequency)}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{fmtIDR(c.monetary)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginTop: 16, gap: 8, fontSize: 12, color: C.muted }}>
                      <button onClick={() => setRfmCurrentPage(p => Math.max(1, p - 1))} disabled={rfmCurrentPage === 1} style={{ background:C.surfaceAlt, color:C.text, border:`1px solid ${C.border}`, borderRadius:4, padding:'4px 10px', cursor:'pointer' }}>Sebelumnya</button>
                      <span>Halaman {rfmCurrentPage} dari {Math.ceil(customers.length / RFM_PAGE_SIZE)}</span>
                      <button onClick={() => setRfmCurrentPage(p => Math.min(Math.ceil(customers.length / RFM_PAGE_SIZE), p + 1))} disabled={rfmCurrentPage === Math.ceil(customers.length / RFM_PAGE_SIZE)} style={{ background:C.surfaceAlt, color:C.text, border:`1px solid ${C.border}`, borderRadius:4, padding:'4px 10px', cursor:'pointer' }}>Berikutnya</button>
                    </div>
                  </div>
                )}
              </Ticket>

              <Ticket style={{ padding:22, marginBottom:24 }}>
                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', flexWrap:'wrap', gap:12, marginBottom:16 }}>
                  <div>
                    <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, color:C.amber, marginBottom:4 }}>LANGKAH 1: EVALUASI MODEL</div>
                    <div style={{ fontFamily:'"Fraunces",serif', fontSize:19, fontWeight:600 }}>Menentukan Jumlah Kelompok (k) Optimal</div>
                    <div style={{ fontSize:13, color:C.muted, marginTop:4, maxWidth: 600 }}>Gunakan metode siku dan skor silhouette untuk membantu menemukan nilai 'k' terbaik untuk data Anda.</div>
                  </div>
                  <button onClick={runEvaluation} disabled={evaluationLoading} style={{ background:C.amber, color:'#1A1306', border:'none', borderRadius:6, padding:'11px 22px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13, fontWeight:600, cursor:'pointer', opacity:evaluationLoading?0.5:1 }}>{evaluationLoading?'Menghitung...':'Jalankan Evaluasi (k=2-8)'}</button>
                </div>
                {evaluationData && (
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap: 24, borderTop: `1px solid ${C.border}`, paddingTop: 20 }}>
                    <div>
                      <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, color:C.muted, marginBottom:4 }}>Metode Siku (WCSS)</div>
                      <p style={{ fontSize:12, color:C.muted, marginTop:0, marginBottom:12, lineHeight:1.6 }}>Pilih 'k' di titik "siku" (penurunan WCSS mulai melandai). Nilai WCSS yang lebih rendah lebih baik.</p>
                      <ResponsiveContainer width="100%" height={240}>
                        <LineChart data={evaluationData}>
                          <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
                          <XAxis dataKey="k" type="number" domain={['dataMin', 'dataMax']} stroke={C.muted} fontSize={11} label={{value:'k (jumlah kelompok)',position:'insideBottom',offset:-2,fill:C.muted,fontSize:11}} />
                          <YAxis stroke={C.muted} fontSize={11} label={{value:'WCSS',angle:-90,position:'insideLeft',fill:C.muted,fontSize:11}} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Line type="monotone" dataKey="wcss" stroke={C.amber} strokeWidth={2.5} dot={{r:4,fill:C.amber}} name="WCSS" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <div>
                      <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, color:C.muted, marginBottom:4 }}>Skor Pemisahan Kelompok</div>
                      <p style={{ fontSize:12, color:C.muted, marginTop:0, marginBottom:12, lineHeight:1.6 }}>Pilih 'k' dengan skor tertinggi. Skor mendekati +1 menunjukkan kelompok pelanggan yang rapat dan terpisah dengan baik.</p>
                      <ResponsiveContainer width="100%" height={240}>
                        <LineChart data={evaluationData}>
                          <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
                          <XAxis dataKey="k" type="number" domain={['dataMin', 'dataMax']} stroke={C.muted} fontSize={11} label={{value:'k (jumlah kelompok)',position:'insideBottom',offset:-2,fill:C.muted,fontSize:11}} />
                          <YAxis stroke={C.muted} fontSize={11} domain={[0, 1]} label={{value:'Skor',angle:-90,position:'insideLeft',fill:C.muted,fontSize:11}} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Line type="monotone" dataKey="silhouette" stroke={C.teal} strokeWidth={2.5} dot={{r:4,fill:C.teal}} name="Skor pemisahan" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}
              </Ticket>

              <Ticket style={{ padding:22, marginBottom:24 }}>
                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:16 }}>
                  <div>
                    <div style={{ fontFamily:'"IBM Plex Mono",monospace', fontSize:12, color:C.amber, marginBottom:4 }}>LANGKAH 2: SEGMENTASI</div>
                    <div style={{ fontFamily:'"Fraunces",serif', fontSize:19, fontWeight:600 }}>Jalankan K-Means</div>
                  </div>
                  <div style={{ display:'flex', alignItems:'center', gap:14 }}>
                    <label style={{ fontSize:13, color:C.muted, display:'flex', alignItems:'center', gap:8 }}>
                      Nilai k Pilihan:
                      <input type="number" min={2} max={Math.min(8,customers.length)} value={kValue} onChange={e=>setKValue(Math.max(2,Math.min(8,Number(e.target.value)||2)))} style={{ width:64, background:C.surfaceAlt, color:C.text, border:`1px solid ${C.border}`, borderRadius:5, padding:'8px 10px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13 }} />
                    </label>
                    <button onClick={runClustering} disabled={clusterLoading} style={{ background:C.amber, color:'#1A1306', border:'none', borderRadius:6, padding:'11px 22px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13, fontWeight:600, cursor:'pointer', opacity:clusterLoading?0.5:1 }}>{clusterLoading?'Memproses...':'Jalankan K-Means'}</button>
                  </div>
                </div>
              </Ticket>

              {clusterSummary && (
                <>
                  <div style={{ display:'grid', gridTemplateColumns:'1.3fr 1fr', gap:18, marginBottom:24 }}>
                    <Ticket style={{ padding:22 }}>
                      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:16, flexWrap:'wrap', gap:10 }}>
                        <div style={{ fontSize:12, color:C.muted, fontFamily:'"IBM Plex Mono",monospace' }}>VISUALISASI KELOMPOK</div>
                        <div style={{ display:'flex', gap:8 }}>
                          {[['Sumbu X',axisX,setAxisX],['Sumbu Y',axisY,setAxisY]].map(([lbl,val,setter])=>(
                            <label key={lbl} style={{ fontSize:11, color:C.muted, display:'flex', alignItems:'center', gap:6 }}>
                              {lbl}
                              <select value={val} onChange={e=>setter(e.target.value)} style={{ background:C.surfaceAlt, color:C.text, border:`1px solid ${C.border}`, borderRadius:5, padding:'6px 8px', fontFamily:'"IBM Plex Mono",monospace', fontSize:11 }}>
                                {AXIS_OPTIONS.map(o=><option key={o.key} value={o.key}>{o.label}</option>)}
                              </select>
                            </label>
                          ))}
                        </div>
                      </div>
                      <ResponsiveContainer width="100%" height={320}>
                        <ScatterChart margin={{left:10,right:10,bottom:10}}>
                          <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
                          <XAxis type="number" dataKey={axisX} name={AXIS_OPTIONS.find(a=>a.key===axisX).label} stroke={C.muted} fontSize={11} tickFormatter={v=>axisX==='monetary'&&v>=1e6?`${(v/1e6).toFixed(0)}jt`:v} />
                          <YAxis type="number" dataKey={axisY} name={AXIS_OPTIONS.find(a=>a.key===axisY).label} stroke={C.muted} fontSize={11} tickFormatter={v=>axisY==='monetary'&&v>=1e6?`${(v/1e6).toFixed(0)}jt`:v} />
                          <ZAxis range={[40,41]} />
                          <Tooltip contentStyle={tooltipStyle} formatter={(v,n)=>n==='monetary'?fmtIDR(v):v} />
                          {clusterSummary.map(s=>(
                            <Scatter key={s.kelompok} name={s.label} data={scatterData.filter(d=>d.kelompok===s.kelompok)} fill={CLUSTER_COLORS[s.kelompok%CLUSTER_COLORS.length]} />
                          ))}
                          <Legend wrapperStyle={{fontSize:11}} />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </Ticket>
                    <Ticket style={{ padding:22 }}>
                      <div style={{ fontSize:12, color:C.muted, marginBottom:16, fontFamily:'"IBM Plex Mono",monospace' }}>PROPORSI SEGMEN</div>
                      <ResponsiveContainer width="100%" height={260}>
                        <PieChart>
                          <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({percent})=>`${(percent*100).toFixed(0)}%`}>
                            {pieData.map((entry,i)=><Cell key={i} fill={CLUSTER_COLORS[entry.kelompok%CLUSTER_COLORS.length]} />)}
                          </Pie>
                          <Tooltip contentStyle={tooltipStyle} />
                        </PieChart>
                      </ResponsiveContainer>
                      <div style={{ display:'flex', flexDirection:'column', gap:6, marginTop:8 }}>
                        {pieData.map((entry,i)=>(
                          <div key={i} style={{ display:'flex', alignItems:'center', gap:8, fontSize:12, color:C.muted }}>
                            <span style={{ width:10, height:10, borderRadius:'50%', background:CLUSTER_COLORS[entry.kelompok%CLUSTER_COLORS.length], display:'inline-block' }} />
                            {entry.name}
                          </div>
                        ))}
                      </div>
                    </Ticket>
                  </div>

                  <Ticket style={{ padding:22, marginBottom:24, overflowX:'auto' }}>
                    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:14, flexWrap:'wrap', gap:10 }}>
                      <div style={{ fontSize:12, color:C.muted, fontFamily:'"IBM Plex Mono",monospace' }}>KARAKTERISTIK KELOMPOK</div>
                      <button onClick={downloadResults} style={{ background:'transparent', color:C.text, border:`1px solid ${C.border}`, borderRadius:6, padding:'10px 20px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13, cursor:'pointer' }}>Unduh Data Per Pelanggan (CSV)</button>
                    </div>
                    <table style={{ width:'100%', borderCollapse:'collapse', fontSize:13 }}>
                      <thead><tr>{['Kelompok','Segmen','Pelanggan','Rata-rata kebaruan','Tingkat kebaruan','Rata-rata frekuensi','Tingkat frekuensi','Rata-rata nilai belanja','Tingkat nilai belanja'].map(h=><th key={h} style={{ textAlign:'left', padding:'10px 14px', color:C.muted, borderBottom:`1px solid ${C.border}`, fontSize:11, textTransform:'uppercase', letterSpacing:'0.08em' }}>{h}</th>)}</tr></thead>
                      <tbody>
                        {clusterInterpretations.map(s=>(
                          <tr key={s.kelompok}>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}><span style={{ display:'inline-block', width:10, height:10, borderRadius:'50%', background:CLUSTER_COLORS[s.kelompok%CLUSTER_COLORS.length], marginRight:8 }} />{s.kelompok}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'Inter,sans-serif', fontWeight:500 }}>{s.label}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{fmtNum(s.count)}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{fmtNum(s.avgR,1)}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}><span style={{ background: s.levelR==='Tinggi'?C.green:s.levelR==='Rendah'?C.rose:C.surfaceAlt, color: s.levelR==='Tinggi'?'#0A1A0A':s.levelR==='Rendah'?'#2A0A0A':C.text, padding:'3px 6px', borderRadius:4, fontSize:11 }}>{s.levelR}</span></td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{fmtNum(s.avgF,1)}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}><span style={{ background: s.levelF==='Tinggi'?C.green:s.levelF==='Rendah'?C.rose:C.surfaceAlt, color: s.levelF==='Tinggi'?'#0A1A0A':s.levelF==='Rendah'?'#2A0A0A':C.text, padding:'3px 6px', borderRadius:4, fontSize:11 }}>{s.levelF}</span></td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}>{fmtIDR(s.avgM)}</td>
                            <td style={{ padding:'10px 14px', borderBottom:`1px solid ${C.border}`, fontFamily:'"IBM Plex Mono",monospace' }}><span style={{ background: s.levelM==='Tinggi'?C.green:s.levelM==='Rendah'?C.rose:C.surfaceAlt, color: s.levelM==='Tinggi'?'#0A1A0A':s.levelM==='Rendah'?'#2A0A0A':C.text, padding:'3px 6px', borderRadius:4, fontSize:11 }}>{s.levelM}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Ticket>

                  {clusterInterpretations && executiveSummary && (
                    <>
                      <Ticket style={{ padding: 22, marginBottom: 24 }}>
                        <div style={{ fontSize:12, color:C.muted, marginBottom:12, fontFamily:'"IBM Plex Mono",monospace', letterSpacing:'0.08em' }}>RINGKASAN EKSEKUTIF</div>
                        <p style={{ fontSize:13.5, color:C.text, lineHeight:1.75, margin:0 }}>
                          Dari hasil segmentasi ini, segmen <strong style={{color:C.amber}}>{executiveSummary.top.label}</strong> memberikan kontribusi pendapatan terbesar yaitu sekitar {fmtNum(executiveSummary.top.pctRevenue,1)}% dari total pendapatan meski hanya berisi {fmtNum(executiveSummary.top.pctCustomers,1)}% dari total pelanggan ({fmtNum(executiveSummary.top.count)} pelanggan), menjadikannya aset utama yang perlu dijaga dan dipertahankan.
                          {executiveSummary.atRisk && (
                            <> Di sisi lain, segmen <strong style={{color:C.rose}}>{executiveSummary.atRisk.label}</strong> berisiko kehilangan pelanggan dengan {fmtNum(executiveSummary.atRisk.count)} pelanggan ({fmtNum(executiveSummary.atRisk.pctCustomers,1)}%) yang sudah lama tidak bertransaksi (rata-rata {fmtNum(executiveSummary.atRisk.avgR,1)} hari sejak transaksi terakhir), sehingga memerlukan perhatian khusus melalui strategi pengaktifan kembali agar tidak hilang sepenuhnya.</>
                          )}
                        </p>
                      </Ticket>

                      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:14, flexWrap:'wrap', gap:10 }}>
                        <div>
                          <div style={{ fontFamily:'"Fraunces",serif', fontSize:19, fontWeight:600 }}>Interpretasi & Rekomendasi per Segmen</div>
                          <div style={{ fontSize:13, color:C.muted, marginTop:4 }}>Penjabaran karakteristik tiap kelompok beserta strategi bisnis dan pemasaran yang relevan.</div>
                        </div>
                        <button onClick={downloadSummaryReport} style={{ background:'transparent', color:C.text, border:`1px solid ${C.border}`, borderRadius:6, padding:'10px 20px', fontFamily:'"IBM Plex Mono",monospace', fontSize:13, cursor:'pointer', flexShrink:0 }}>Unduh Ringkasan dan Rekomendasi (CSV)</button>
                      </div>

                      <div style={{ display:'flex', flexDirection:'column', gap:16, marginBottom:24 }}>
                        {clusterInterpretations.map(s => (
                          <Ticket key={s.kelompok} style={{ padding:22 }}>
                            <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:12, flexWrap:'wrap' }}>
                              <span style={{ width:12, height:12, borderRadius:'50%', background:CLUSTER_COLORS[s.kelompok%CLUSTER_COLORS.length], display:'inline-block', flexShrink:0 }} />
                              <div style={{ fontFamily:'"Fraunces",serif', fontSize:18, fontWeight:600 }}>{s.label}</div>
                              <div style={{ fontSize:12, color:C.muted, fontFamily:'"IBM Plex Mono",monospace' }}>
                                {fmtNum(s.count)} pelanggan - {fmtNum(s.pctCustomers,1)}% populasi - {fmtNum(s.pctRevenue,1)}% kontribusi pendapatan
                              </div>
                            </div>
                            <p style={{ fontSize:13.5, color:C.text, lineHeight:1.75, marginBottom:14 }}>{s.narrative}</p>
                            <div style={{ fontSize:11, color:C.muted, textTransform:'uppercase', letterSpacing:'0.08em', marginBottom:8, fontFamily:'"IBM Plex Mono",monospace' }}>Rekomendasi Strategi Bisnis dan Pemasaran</div>
                            <ul style={{ margin:0, paddingLeft:18, display:'flex', flexDirection:'column', gap:6 }}>
                              {s.recommendations.map((r,i) => <li key={i} style={{ fontSize:13.5, color:C.text, lineHeight:1.7 }}>{r}</li>)}
                            </ul>
                          </Ticket>
                        ))}
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          )
        )}
      </main>
    </div>
  );
}
