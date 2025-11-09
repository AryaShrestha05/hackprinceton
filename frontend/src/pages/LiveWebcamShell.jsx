import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import SubtleDots from '@/components/ui/subtle-dots'
import SmokeyCursor from '@/components/ui/smokey-cursor'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000'

export default function LiveWebcamShell() {
  const rootRef = useRef(null)
  const [streamError, setStreamError] = useState('')

  useEffect(() => {
    const el = rootRef.current
    if (!el) return
    gsap.fromTo(
      el,
      { opacity: 0 },
      { opacity: 1, duration: 0.45, ease: 'power1.out' }
    )
  }, [])

  return (
    <div ref={rootRef} className="relative min-h-screen overflow-hidden bg-aurora">
      <SubtleDots />
      <SmokeyCursor
        simulationResolution={96}
        dyeResolution={960}
        densityDissipation={4.5}
        velocityDissipation={2.5}
        curl={2}
        splatRadius={0.12}
        splatForce={3200}
        colorUpdateSpeed={6}
      />

      <div className="pointer-events-none absolute inset-0">
        <div className="grid-overlay absolute inset-0" aria-hidden="true" />
      </div>
      <div className="relative z-10 flex min-h-screen flex-col gap-8 px-6 py-10 md:px-12 lg:px-16">
        <div className="flex flex-1 items-center justify-center">
          <section className="webcam-shell">
            <div className="webcam-shell__placeholder">
              {streamError ? (
                <div className="webcam-shell__error">
                  <h1>stream unavailable</h1>
                  <p>{streamError}</p>
                </div>
              ) : (
                <img
                  className="webcam-shell__video"
                  src={`${API_BASE_URL}/api/video_feed`}
                  alt="live posture stream"
                  onError={() => setStreamError('unable to load the webcam stream. is the flask backend running?')}
                />
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

