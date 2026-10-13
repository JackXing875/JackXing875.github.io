// Small, optional enhancements; reading and citations also work without JavaScript.
document.querySelectorAll('.copy-citation').forEach((button) => {
  if (!navigator.clipboard) return;
  button.hidden = false;
  button.addEventListener('click', async () => {
    const citation = button.closest('.citation-content');
    const status = citation.querySelector('.citation-status');
    try {
      await navigator.clipboard.writeText(citation.querySelector('code').textContent.trim());
      status.textContent = 'Citation copied.';
    } catch (error) {
      status.textContent = 'Select and copy the citation below.';
    }
  });
});

const contents = document.querySelector('.reading-toc');
if (contents) {
  const headings = [...document.querySelectorAll('.reading-content h2, .reading-content h3')]
    .filter(heading => heading.textContent.trim());
  if (headings.length > 1) {
    const list = document.createElement('ol');
    let section;
    const links = new Map();
    headings.forEach((heading, index) => {
      if (!heading.id) heading.id = 'section-' + (index + 1);
      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = '#' + encodeURIComponent(heading.id);
      link.textContent = heading.textContent.trim();
      item.append(link);
      if (heading.tagName === 'H2' || !section) {
        list.append(item);
        if (heading.tagName === 'H2') section = item;
      } else {
        let children = section.querySelector('ol');
        if (!children) {
          children = document.createElement('ol');
          section.append(children);
        }
        children.append(item);
      }
      links.set(heading, link);
    });
    contents.querySelector('nav').append(list);
    // Native anchors handle Unicode IDs and respect the sticky header's scroll margin.
    // Keep the theme's older jQuery smooth-scroll handler out of these clicks.
    contents.addEventListener('click', (event) => {
      if (event.target.closest('a')) event.stopPropagation();
    }, true);
    contents.hidden = false;
    const desktop = window.matchMedia('(min-width: 1100px)');
    const setExpanded = () => { contents.querySelector('details').open = desktop.matches; };
    setExpanded();
    desktop.addEventListener('change', setExpanded);

    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        const visible = entries.filter(entry => entry.isIntersecting);
        if (!visible.length) return;
        links.forEach(link => link.removeAttribute('aria-current'));
        links.get(visible[0].target).setAttribute('aria-current', 'location');
      }, {rootMargin: '-125px 0px -60% 0px'});
      headings.forEach(heading => observer.observe(heading));
    }
  }
}
