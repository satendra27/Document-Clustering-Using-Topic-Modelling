const headers = document.querySelectorAll('.accordion-header');

headers.forEach(header => {
  header.addEventListener('click', () => {
    const content = header.nextElementSibling;

    header.classList.toggle('active');
    content.classList.toggle('show');

    // Change the icon
    if (header.textContent.trim().startsWith('➕')) {
      header.textContent = header.textContent.replace('➕', '➖');
    } else {
      header.textContent = header.textContent.replace('➖', '➕');
    }
  });
});
