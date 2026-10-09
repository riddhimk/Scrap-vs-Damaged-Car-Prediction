document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const dropZoneContent = document.getElementById('dropZoneContent');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const clearBtn = document.getElementById('clearBtn');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');

    if (!dropZone) return; // Only run on index page

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFile(files[0]);
        }
    });

    // Handle selected file via button
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    function handleFile(file) {
        // Basic validation
        if (!file.type.match('image/jpeg') && !file.type.match('image/png')) {
            alert('Please upload a valid JPG or PNG image.');
            clearFile();
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File is too large. Maximum size is 10MB.');
            clearFile();
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZoneContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearFile() {
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropZoneContent.classList.remove('hidden');
        submitBtn.disabled = true;
    }

    clearBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation(); // prevent clicking dropzone underneath
        clearFile();
    });

    uploadForm.addEventListener('submit', () => {
        if (!submitBtn.disabled) {
            btnText.textContent = 'Analyzing...';
            btnLoader.classList.remove('hidden');
            submitBtn.disabled = true; // Prevent double submission
            
            // Allow form to submit by not preventing default
        }
    });
});
