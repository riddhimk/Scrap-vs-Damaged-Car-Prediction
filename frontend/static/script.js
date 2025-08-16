document.getElementById("fileInput").addEventListener("change", function (e) {
  const file = e.target.files[0];
  const reader = new FileReader();
  
  reader.onload = function (event) {
    const imgPreview = `
      <div class="image-preview">
        <img src="${event.target.result}" alt="Preview Image"/>
        <button class="remove-btn" onclick="clearImage()"><i class="fas fa-times"></i></button>
      </div>`;
    document.getElementById("previewScroll").innerHTML = imgPreview;
  };

  if (file) {
    reader.readAsDataURL(file);
  }
});

function clearImage() {
  const fileInput = document.getElementById("fileInput");
  fileInput.value = '';
  document.getElementById("previewScroll").innerHTML = '';
}