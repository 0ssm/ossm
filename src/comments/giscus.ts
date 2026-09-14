const config = {
  repo: '0ssm/ossm',
  repoId: 'R_kgDOUXelxw',
  category: 'Comments',
  categoryId: 'DIC_kwDOUXelx84DFkbB',
  mapping: 'pathname',
  reactionsEnabled: '1',
  inputPosition: 'top',
  theme: 'preferred_color_scheme',
  lang: 'en',
}

const GISCUS_ORIGIN = 'https://giscus.app'

const isDark = () => document.documentElement.classList.contains('dark')

const giscusTheme = (dark: boolean) => (dark ? 'dark' : 'light')

export function mountComment(container: HTMLElement): void {
  const script = document.createElement('script')

  script.src = `${GISCUS_ORIGIN}/client.js`
  script.async = true
  script.crossOrigin = 'anonymous'

  script.setAttribute('data-repo', config.repo)
  script.setAttribute('data-repo-id', config.repoId)
  script.setAttribute('data-category', config.category)
  script.setAttribute('data-category-id', config.categoryId)
  script.setAttribute('data-mapping', config.mapping)
  script.setAttribute('data-strict', '0')
  script.setAttribute('data-reactions-enabled', config.reactionsEnabled)
  script.setAttribute('data-emit-metadata', '0')
  script.setAttribute('data-input-position', config.inputPosition)
  script.setAttribute('data-theme', giscusTheme(isDark()))
  script.setAttribute('data-lang', config.lang)
  script.setAttribute('data-loading', 'lazy')

  container.appendChild(script)
}

export function onThemeChange(dark: boolean): void {
  const iframe = document.querySelector<HTMLIFrameElement>(
    'iframe.giscus-frame',
  )

  iframe?.contentWindow?.postMessage(
    {
      giscus: {
        setConfig: {
          theme: giscusTheme(dark),
        },
      },
    },
    GISCUS_ORIGIN,
  )
}