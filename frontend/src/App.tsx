import { useEffect, useRef, useState } from 'react'

const MAX_BYTES = 10 * 1024 * 1024
const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp']

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [original, setOriginal] = useState('')
  const [result, setResult] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [ready, setReady] = useState<boolean | null>(null)
  const [split, setSplit] = useState(50)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(data => setReady(Boolean(data.model_available))).catch(() => setReady(false))
  }, [])

  useEffect(() => {
    if (!file) { setOriginal(''); return }
    const url = URL.createObjectURL(file)
    setOriginal(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => () => { if (result) URL.revokeObjectURL(result) }, [result])

  function selectFile(next?: File) {
    if (!next) return
    setError('')
    if (!ACCEPTED.includes(next.type)) { setError('Choose a JPEG, PNG, or WebP image.'); return }
    if (next.size > MAX_BYTES) { setError('Maximum image size is 10 MB.'); return }
    setResult(previous => { if (previous) URL.revokeObjectURL(previous); return '' })
    setFile(next)
    setSplit(50)
  }

  async function enhance() {
    if (!file || busy) return
    setBusy(true); setError('')
    try {
      const body = new FormData()
      body.append('file', file)
      const response = await fetch('/api/enhance', { method: 'POST', body })
      if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(typeof data.detail === 'string' ? data.detail : 'Enhancement failed.')
      }
      const blob = await response.blob()
      setResult(previous => { if (previous) URL.revokeObjectURL(previous); return URL.createObjectURL(blob) })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally { setBusy(false) }
  }

  return <div className="page">
    <header className="topbar"><span className="brand"><span className="mark">✦</span> Restore<span className="brand-light">Studio</span></span><span className="version">IMAGE RESTORATION LAB · V0.1</span></header>
    <main>
      <section className="intro"><span className="eyebrow">A LITTLE DETAIL GOES A LONG WAY</span><h1>Bring the details <em>back.</em></h1><p>Explore neural image restoration with a real, benchmarked model. Upload a photo, enhance it, and compare the difference.</p></section>
      <div className="workspace">
        <section className="panel controls"><div className="panel-heading"><span className="step">01</span><h2>Your image</h2></div>
          <input ref={inputRef} type="file" accept="image/png,image/jpeg,image/webp" className="sr-only" aria-label="Select image" onChange={event => selectFile(event.target.files?.[0])} />
          <div className="dropzone" role="button" tabIndex={0} onClick={() => inputRef.current?.click()} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inputRef.current?.click() } }} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); selectFile(event.dataTransfer.files[0]) }}>
            <span className="upload-icon">↑</span><strong>{file ? file.name : 'Drop your image here'}</strong><span>{file ? (file.size / 1024 / 1024).toFixed(2) + ' MB · Click to replace' : 'or click to browse your files'}</span><small>PNG, JPG or WebP · Up to 10 MB</small>
          </div>
          <div className="model"><span className="model-dot" /><div><strong>Residual restoration CNN</strong><small>{ready === null ? 'Checking model availability…' : ready ? 'Model ready for inference' : 'Model unavailable · See setup instructions'}</small></div></div>
          <button className="primary" disabled={!file || busy || ready !== true} onClick={enhance}>{busy ? 'Enhancing image…' : '✦ Enhance image'}</button>
          {error && <p className="error" role="alert">{error}</p>}
          <p className="hint">Optimized for 80 × 80 face-image restoration. The output is 80 × 80, not full-resolution upscaling.</p>
        </section>
        <section className="panel preview"><div className="panel-heading"><span className="step">02</span><h2>Compare results</h2><span className="panel-caption">BEFORE / AFTER</span></div>
          <div className="canvas">
            {original ? result ? <div className="comparison" style={{aspectRatio: '1 / 1'}}>
              <img src={original} alt="Original uploaded image" className="comparison-image" />
              <div className="after-layer" style={{clipPath: `inset(0 ${100 - split}% 0 0)`}}><img src={result} alt="Restored image" className="comparison-image" /></div>
              <div className="divider-line" style={{left: `${split}%`}}><span>↔</span></div>
              <input className="comparison-range" type="range" min="0" max="100" value={split} onChange={e => setSplit(Number(e.target.value))} aria-label="Reveal enhanced image; left is enhanced, right is original" />
              <span className="image-label label-before">AFTER</span><span className="image-label label-after">BEFORE</span>
            </div> : <img className="single-image" src={original} alt="Selected image preview" /> : <div className="empty"><span className="empty-symbol">▧</span><strong>Your transformation starts here</strong><span>Upload an image to preview the results</span></div>}
          </div>
          <div className="preview-footer"><span>{result ? 'Drag the slider to compare' : 'Your preview will appear here'}</span>{result && <a className="download" href={result} download="enhanced.png">↓ Download PNG</a>}</div>
        </section>
      </div>
      <footer><span>BUILT WITH REACT + FASTAPI + TENSORFLOW</span><span>REAL METRICS. REAL IMAGES. NO MAGIC.</span></footer>
    </main>
  </div>
}
