/**
 * main.js - Hệ thống tương tác Vị Bắc (Chuẩn Figma 100%)
 * Quản lý đa trạng thái: Tải ảnh kéo thả -> Preview sẵn sàng -> Spinner loading -> Kết quả 2 cột / Độ tin cậy thấp / Báo lỗi.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Containers
    const stateUpload = document.getElementById('state-upload');
    const stateSelected = document.getElementById('state-selected');
    const stateLoading = document.getElementById('state-loading');
    const stateSuccess = document.getElementById('state-result-success');
    const stateLowConf = document.getElementById('state-result-low-conf');
    const stateError = document.getElementById('state-error');

    // Controls
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('dishFileInput');
    const btnBrowse = document.getElementById('btnBrowseFile');
    const btnStartPredict = document.getElementById('btnStartPredict');
    const btnChangeImage = document.getElementById('btnChangeImage');
    const btnResetUpload = document.getElementById('btnResetUpload');
    const btnResetLow = document.getElementById('btnResetLow');
    const btnTryAgain = document.getElementById('btnTryAgain');
    const btnResetError = document.getElementById('btnResetError');
    const editorCanvas = document.getElementById('editorCanvas');
    const btnCropImage = document.getElementById('btnCropImage');
    const btnRotateLeft = document.getElementById('btnRotateLeft');
    const btnRotateRight = document.getElementById('btnRotateRight');
    const btnResetEdit = document.getElementById('btnResetEdit');
    const editorStatus = document.getElementById('editorStatus');

    // Previews
    const loadingImgPreview = document.getElementById('loadingImagePreview');
    const selectedFileName = document.getElementById('selectedFileName');
    const selectedFileSize = document.getElementById('selectedFileSize');

    let currentFile = null;
    let originalFile = null;
    let currentPreviewUrl = '';
    let editorImage = null;
    let selectionRect = null;
    let dragStart = null;
    const sourceCanvas = document.createElement('canvas');
    const sourceContext = sourceCanvas.getContext('2d');
    const editorContext = editorCanvas ? editorCanvas.getContext('2d') : null;

    // Switch states
    function showState(targetState) {
        const allStates = [stateUpload, stateSelected, stateLoading, stateSuccess, stateLowConf, stateError];
        allStates.forEach(st => {
            if (st) st.style.display = 'none';
        });
        if (targetState) {
            targetState.style.display = 'block';
        }
    }

    function formatFileSize(bytes) {
        if (!bytes) return '';
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1).replace('.', ',') + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1).replace('.', ',') + ' MB';
    }

    function triggerError(title, desc) {
        const errTitle = document.getElementById('errTitle');
        const errDesc = document.getElementById('errDesc');
        if (errTitle) errTitle.textContent = title;
        if (errDesc) errDesc.textContent = desc;
        showState(stateError);
    }

    function renderEditor() {
        if (!editorContext || !sourceCanvas.width || !sourceCanvas.height) return;
        editorCanvas.width = sourceCanvas.width;
        editorCanvas.height = sourceCanvas.height;
        editorContext.clearRect(0, 0, editorCanvas.width, editorCanvas.height);
        editorContext.drawImage(sourceCanvas, 0, 0);

        if (selectionRect) {
            const { x, y, width, height } = selectionRect;
            editorContext.fillStyle = 'rgba(0, 0, 0, 0.45)';
            editorContext.fillRect(0, 0, editorCanvas.width, y);
            editorContext.fillRect(0, y + height, editorCanvas.width, editorCanvas.height - y - height);
            editorContext.fillRect(0, y, x, height);
            editorContext.fillRect(x + width, y, editorCanvas.width - x - width, height);
            editorContext.save();
            editorContext.setLineDash([8, 5]);
            editorContext.lineWidth = Math.max(2, editorCanvas.width / 500);
            editorContext.strokeStyle = '#fff';
            editorContext.strokeRect(x, y, width, height);
            editorContext.restore();
        }
    }

    function loadEditorImage(file, isOriginal) {
        const nextUrl = URL.createObjectURL(file);
        const previousUrl = currentPreviewUrl;
        const image = new Image();
        image.onload = () => {
            if (currentPreviewUrl !== nextUrl) {
                URL.revokeObjectURL(nextUrl);
                return;
            }

            editorImage = image;
            const scale = Math.min(1, 1800 / Math.max(image.naturalWidth, image.naturalHeight));
            sourceCanvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
            sourceCanvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
            sourceContext.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
            sourceContext.drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height);
            selectionRect = null;
            renderEditor();
            if (loadingImgPreview) loadingImgPreview.src = nextUrl;
            if (selectedFileName) selectedFileName.textContent = currentFile.name;
            if (selectedFileSize) selectedFileSize.textContent = formatFileSize(currentFile.size);
            if (editorStatus) editorStatus.textContent = 'Kéo trên ảnh để chọn vùng cần giữ lại.';
            if (btnCropImage) btnCropImage.disabled = true;
            showState(stateSelected);
            if (previousUrl) URL.revokeObjectURL(previousUrl);
        };
        image.onerror = () => {
            URL.revokeObjectURL(nextUrl);
            currentPreviewUrl = '';
            triggerError('Không thể mở ảnh', 'Tệp ảnh bị lỗi hoặc trình duyệt không thể đọc được ảnh này.');
        };

        currentFile = file;
        if (isOriginal) originalFile = file;
        currentPreviewUrl = nextUrl;
        image.src = nextUrl;
    }

    function handleSelectedFile(file) {
        if (!file) return;

        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            triggerError(
                'File không phải ảnh',
                'Định dạng tệp không được hỗ trợ. Vui lòng tải lên định dạng JPG, PNG hoặc WebP.'
            );
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            triggerError(
                'Ảnh quá lớn (tối đa 5MB)',
                'Dung lượng ảnh vượt quá 5MB. Vui lòng nén ảnh hoặc chọn ảnh khác nhỏ hơn.'
            );
            return;
        }

        loadEditorImage(file, true);
    }

    function canvasToJpegFile(canvas, name) {
        return new Promise((resolve, reject) => {
            canvas.toBlob((blob) => {
                if (!blob) {
                    reject(new Error('Không thể xử lý ảnh trên trình duyệt.'));
                    return;
                }
                if (blob.size > 5 * 1024 * 1024) {
                    reject(new Error('Ảnh sau khi chỉnh sửa vượt quá giới hạn 5MB.'));
                    return;
                }
                const baseName = name.replace(/\.[^.]+$/, '') || 'anh-mon-an';
                resolve(new File([blob], `${baseName}.jpg`, {
                    type: 'image/jpeg',
                    lastModified: Date.now()
                }));
            }, 'image/jpeg', 0.9);
        });
    }

    async function applyEditedCanvas(canvas) {
        if (!currentFile) return;
        try {
            const editedFile = await canvasToJpegFile(canvas, currentFile.name);
            loadEditorImage(editedFile, false);
        } catch (error) {
            if (editorStatus) editorStatus.textContent = error.message;
        }
    }

    // Browse click
    if (btnBrowse && fileInput) {
        btnBrowse.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    if (dropArea && fileInput) {
        dropArea.addEventListener('click', (e) => {
            if (e.target !== btnBrowse && !btnBrowse.contains(e.target)) {
                fileInput.click();
            }
        });

        // Drag & Drop
        ['dragenter', 'dragover'].forEach(name => {
            dropArea.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            dropArea.addEventListener(name, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropArea.classList.remove('dragover');
            });
        });

        dropArea.addEventListener('drop', (e) => {
            if (e.dataTransfer && e.dataTransfer.files.length > 0) {
                handleSelectedFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleSelectedFile(e.target.files[0]);
            }
        });
    }

    function canvasPoint(event) {
        const bounds = editorCanvas.getBoundingClientRect();
        return {
            x: Math.max(0, Math.min(editorCanvas.width, (event.clientX - bounds.left) * editorCanvas.width / bounds.width)),
            y: Math.max(0, Math.min(editorCanvas.height, (event.clientY - bounds.top) * editorCanvas.height / bounds.height))
        };
    }

    if (editorCanvas) {
        editorCanvas.addEventListener('pointerdown', (event) => {
            if (!editorImage) return;
            event.preventDefault();
            dragStart = canvasPoint(event);
            selectionRect = { x: dragStart.x, y: dragStart.y, width: 0, height: 0 };
            editorCanvas.setPointerCapture(event.pointerId);
            renderEditor();
        });

        editorCanvas.addEventListener('pointermove', (event) => {
            if (!dragStart) return;
            const point = canvasPoint(event);
            selectionRect = {
                x: Math.min(dragStart.x, point.x),
                y: Math.min(dragStart.y, point.y),
                width: Math.abs(point.x - dragStart.x),
                height: Math.abs(point.y - dragStart.y)
            };
            renderEditor();
        });

        editorCanvas.addEventListener('pointerup', () => {
            if (!dragStart) return;
            dragStart = null;
            const hasSelection = selectionRect && selectionRect.width >= 10 && selectionRect.height >= 10;
            if (!hasSelection) selectionRect = null;
            if (btnCropImage) btnCropImage.disabled = !hasSelection;
            if (editorStatus) {
                editorStatus.textContent = hasSelection
                    ? 'Đã chọn vùng ảnh. Nhấn “Áp dụng cắt” để sử dụng vùng này.'
                    : 'Vùng chọn quá nhỏ. Hãy kéo để chọn vùng ảnh lớn hơn.';
            }
            renderEditor();
        });

        editorCanvas.addEventListener('pointercancel', () => {
            dragStart = null;
            selectionRect = null;
            if (btnCropImage) btnCropImage.disabled = true;
            renderEditor();
        });
    }

    if (btnCropImage) {
        btnCropImage.addEventListener('click', () => {
            if (!selectionRect || !sourceCanvas.width) return;
            const cropCanvas = document.createElement('canvas');
            cropCanvas.width = Math.round(selectionRect.width);
            cropCanvas.height = Math.round(selectionRect.height);
            const cropContext = cropCanvas.getContext('2d');
            cropContext.drawImage(
                sourceCanvas,
                selectionRect.x, selectionRect.y, selectionRect.width, selectionRect.height,
                0, 0, cropCanvas.width, cropCanvas.height
            );
            applyEditedCanvas(cropCanvas);
        });
    }

    function rotateImage(direction) {
        if (!sourceCanvas.width) return;
        const rotatedCanvas = document.createElement('canvas');
        rotatedCanvas.width = sourceCanvas.height;
        rotatedCanvas.height = sourceCanvas.width;
        const context = rotatedCanvas.getContext('2d');
        if (direction === 'left') {
            context.translate(0, rotatedCanvas.height);
            context.rotate(-Math.PI / 2);
        } else {
            context.translate(rotatedCanvas.width, 0);
            context.rotate(Math.PI / 2);
        }
        context.drawImage(sourceCanvas, 0, 0);
        applyEditedCanvas(rotatedCanvas);
    }

    if (btnRotateLeft) btnRotateLeft.addEventListener('click', () => rotateImage('left'));
    if (btnRotateRight) btnRotateRight.addEventListener('click', () => rotateImage('right'));
    if (btnResetEdit) {
        btnResetEdit.addEventListener('click', () => {
            if (originalFile) loadEditorImage(originalFile, true);
        });
    }

    // Reset buttons
    [btnChangeImage, btnResetUpload, btnResetLow, btnTryAgain, btnResetError].forEach(btn => {
        if (btn) {
            btn.addEventListener('click', () => {
                currentFile = null;
                originalFile = null;
                editorImage = null;
                selectionRect = null;
                sourceCanvas.width = 0;
                sourceCanvas.height = 0;
                if (currentPreviewUrl) URL.revokeObjectURL(currentPreviewUrl);
                currentPreviewUrl = '';
                if (fileInput) fileInput.value = '';
                showState(stateUpload);
            });
        }
    });

    // Start Prediction
    if (btnStartPredict) {
        btnStartPredict.addEventListener('click', async () => {
            if (!currentFile) return;

            showState(stateLoading);

            const formData = new FormData();
            formData.append('file', currentFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    triggerError('Lỗi xử lý hình ảnh', data.error || 'Hệ thống gặp sự cố trong quá trình phân tích ảnh.');
                    return;
                }

                renderPredictionResult(data);

            } catch (err) {
                console.error('Lỗi API predict:', err);
                triggerError('Lỗi kết nối máy chủ', 'Không thể gửi yêu cầu nhận diện đến máy chủ. Vui lòng thử lại sau.');
            }
        });
    }

    function renderPredictionResult(data) {
        const isConfident = data.is_confident !== false && (data.confidence >= 0.50);

        if (isConfident) {
            // === RENDER KẾT QUẢ THÀNH CÔNG (2 CỘT FIGMA) ===
            const resUserImg = document.getElementById('resUserImg');
            const resUserFileName = document.getElementById('resUserFileName');
            const resUserFileSize = document.getElementById('resUserFileSize');
            const resMainDishName = document.getElementById('resMainDishName');
            const resConfidenceNum = document.getElementById('resConfidenceNum');
            const resMainBarFill = document.getElementById('resMainBarFill');

            if (resUserImg) resUserImg.src = currentPreviewUrl;
            if (resUserFileName) resUserFileName.textContent = currentFile ? currentFile.name : 'anh-mon-an.jpg';
            if (resUserFileSize) resUserFileSize.textContent = currentFile ? formatFileSize(currentFile.size) : '';
            
            const confPct = Math.round(data.confidence * 100);
            if (resMainDishName) resMainDishName.textContent = data.dish_name;
            if (resConfidenceNum) resConfidenceNum.textContent = confPct + '%';
            if (resMainBarFill) resMainBarFill.style.width = confPct + '%';

            // Top 3 Breakdown
            if (data.top3 && data.top3.length > 0) {
                const top1 = data.top3[0];
                const top1Name = document.getElementById('top1NameMini');
                const top1Pct = document.getElementById('top1PctMini');
                const top1Bar = document.getElementById('top1BarMini');
                if (top1Name) top1Name.textContent = top1.dish;
                if (top1Pct) top1Pct.textContent = top1.confidence_pct;
                if (top1Bar) top1Bar.style.width = top1.confidence_pct;
            }

            if (data.top3 && data.top3.length > 1) {
                const top2 = data.top3[1];
                const top2Name = document.getElementById('top2Name');
                const top2Pct = document.getElementById('top2Pct');
                const top2Bar = document.getElementById('top2Bar');
                if (top2Name) top2Name.textContent = top2.dish;
                if (top2Pct) top2Pct.textContent = top2.confidence_pct;
                if (top2Bar) top2Bar.style.width = top2.confidence_pct;
            }

            if (data.top3 && data.top3.length > 2) {
                const top3 = data.top3[2];
                const top3Name = document.getElementById('top3Name');
                const top3Pct = document.getElementById('top3Pct');
                const top3Bar = document.getElementById('top3Bar');
                if (top3Name) top3Name.textContent = top3.dish;
                if (top3Pct) top3Pct.textContent = top3.confidence_pct;
                if (top3Bar) top3Bar.style.width = top3.confidence_pct;
            }

            // Chi tiết văn hóa món ăn ở Cột phải
            const dishInfo = data.dish_info || {};
            const resDishCoverImg = document.getElementById('resDishCoverImg');
            const resDetailTitle = document.getElementById('resDetailTitle');
            const resDishOrigin = document.getElementById('resDishOrigin');
            const resDishDesc = document.getElementById('resDishDesc');
            const resIngredientsList = document.getElementById('resIngredientsList');
            const resServingGuide = document.getElementById('resServingGuide');
            const resCalorieVal = document.getElementById('resCalorieVal');
            const resDishTypeTag = document.getElementById('resDishTypeTag');

            if (resDetailTitle) resDetailTitle.textContent = data.dish_name;
            if (resDishOrigin) resDishOrigin.innerHTML = `<i class="bi bi-geo-alt"></i> Nguồn gốc: ${dishInfo.nguon_goc || dishInfo.origin || 'Đồng bằng Bắc Bộ'}`;
            if (resDishDesc) resDishDesc.textContent = dishInfo.mo_ta || dishInfo.description || 'Món ăn truyền thống đậm đà bản sắc vùng đất Bắc.';
            if (resServingGuide) resServingGuide.textContent = dishInfo.cach_thuong_thuc || dishInfo.eating_guide || 'Thưởng thức khi còn nóng kèm các loại gia vị đặc trưng.';
            if (resCalorieVal) resCalorieVal.textContent = dishInfo.calo_uoc_tinh || dishInfo.calories || '400 - 500 kcal';

            if (resDishCoverImg) {
                const cleanKey = data.dish_key || data.dish_name;
                resDishCoverImg.src = `/static/img/dishes/${cleanKey}.jpg`;
                resDishCoverImg.onerror = () => {
                    resDishCoverImg.src = currentPreviewUrl;
                };
            }

            if (resDishTypeTag) {
                const typeMap = {
                    'Pho': 'Món nước', 'Bun thang': 'Món nước', 'Bun rieu': 'Món nước', 'Banh da cua': 'Món nước', 'Chao long': 'Món cháo',
                    'Bun cha': 'Món nướng', 'Cha ca La Vong': 'Món chiên', 'Banh tom Ho Tay': 'Món chiên', 'Bun dau mam tom': 'Món chấm',
                    'Banh cuon': 'Món hấp', 'Banh gio': 'Món hấp', 'Banh chung': 'Món bánh', 'Banh duc': 'Món bánh', 'Banh gai': 'Bánh ngọt',
                    'Banh mi': 'Bánh mì', 'Xoi xeo': 'Món xôi', 'Com rang dua bo': 'Món rang', 'Nem chua': 'Món đặc sản'
                };
                resDishTypeTag.textContent = typeMap[data.dish_key] || 'Đặc sản miền Bắc';
            }

            if (resIngredientsList) {
                resIngredientsList.innerHTML = '';
                const ings = dishInfo.nguyen_lieu || dishInfo.ingredients || ['Gia vị truyền thống'];
                const list = Array.isArray(ings) ? ings : ings.split(',');
                list.forEach(item => {
                    const pill = document.createElement('span');
                    pill.className = 'ingredient-pill';
                    pill.textContent = item.trim();
                    resIngredientsList.appendChild(pill);
                });
            }

            showState(stateSuccess);

        } else {
            // === RENDER TRẠNG THÁI CẦN GÓC NHÌN RÕ HƠN ===
            const lowUserImg = document.getElementById('lowUserImg');
            const lowMainDishName = document.getElementById('lowMainDishName');
            const lowConfidenceNum = document.getElementById('lowConfidenceNum');
            const lowMainBarFill = document.getElementById('lowMainBarFill');
            const lowTop3ListContainer = document.getElementById('lowTop3ListContainer');

            if (lowUserImg) lowUserImg.src = currentPreviewUrl;
            const topPct = Math.round(data.confidence * 100);
            if (lowMainDishName) lowMainDishName.textContent = data.dish_name || 'Chưa rõ món';
            if (lowConfidenceNum) lowConfidenceNum.textContent = topPct + '%';
            if (lowMainBarFill) lowMainBarFill.style.width = topPct + '%';

            if (lowTop3ListContainer && data.top3) {
                lowTop3ListContainer.innerHTML = '';
                data.top3.forEach(item => {
                    const row = document.createElement('div');
                    row.className = 'top3-row-item';
                    const labels = document.createElement('div');
                    labels.className = 'top3-labels-row';
                    const dishName = document.createElement('span');
                    dishName.textContent = item.dish;
                    const percentage = document.createElement('span');
                    percentage.textContent = item.confidence_pct;
                    labels.append(dishName, percentage);

                    const track = document.createElement('div');
                    track.className = 'top3-sub-track';
                    const fill = document.createElement('div');
                    fill.className = 'top3-sub-fill';
                    fill.style.width = item.confidence_pct;
                    track.appendChild(fill);
                    row.append(labels, track);
                    lowTop3ListContainer.appendChild(row);
                });
            }

            const dishInfo = data.dish_info || {};
            const lowDetailTitle = document.getElementById('lowDetailTitle');
            const lowDishOrigin = document.getElementById('lowDishOrigin');
            const lowDishDesc = document.getElementById('lowDishDesc');
            const lowIngredientsList = document.getElementById('lowIngredientsList');
            const lowServingGuide = document.getElementById('lowServingGuide');
            const lowCalorieVal = document.getElementById('lowCalorieVal');
            if (lowDetailTitle) lowDetailTitle.textContent = data.dish_name || 'Món ăn được dự đoán';
            if (lowDishOrigin) lowDishOrigin.textContent = `Nguồn gốc: ${dishInfo.nguon_goc || dishInfo.origin || 'Đồng bằng Bắc Bộ'}`;
            if (lowDishDesc) lowDishDesc.textContent = dishInfo.mo_ta || dishInfo.description || 'Món ăn truyền thống đậm đà bản sắc vùng đất Bắc.';
            if (lowServingGuide) lowServingGuide.textContent = dishInfo.cach_thuong_thuc || dishInfo.eating_guide || 'Thưởng thức khi còn nóng kèm các loại gia vị đặc trưng.';
            if (lowCalorieVal) lowCalorieVal.textContent = dishInfo.calo_uoc_tinh || dishInfo.calories || '400 - 500 kcal';
            if (lowIngredientsList) {
                lowIngredientsList.replaceChildren();
                const ingredients = dishInfo.nguyen_lieu || dishInfo.ingredients || ['Gia vị truyền thống'];
                const list = Array.isArray(ingredients) ? ingredients : ingredients.split(',');
                list.forEach((ingredient) => {
                    const pill = document.createElement('span');
                    pill.className = 'ingredient-pill';
                    pill.textContent = ingredient.trim();
                    lowIngredientsList.appendChild(pill);
                });
            }

            showState(stateLowConf);
        }
    }

    // =========================================================================
    // XỬ LÝ TRANG DANH SÁCH MÓN (/dishes)
    // =========================================================================
    const searchDishInput = document.getElementById('searchDishInput');
    const filterPills = document.querySelectorAll('.filter-btn-pill');
    const dishCards = document.querySelectorAll('.card-dish-full');

    if (searchDishInput) {
        searchDishInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            dishCards.forEach(card => {
                const title = card.getAttribute('data-name')?.toLowerCase() || '';
                const desc = card.getAttribute('data-desc')?.toLowerCase() || '';
                const category = card.getAttribute('data-category')?.toLowerCase() || '';
                if (title.includes(query) || desc.includes(query) || category.includes(query)) {
                    card.style.display = 'flex';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }

    filterPills.forEach(pill => {
        pill.addEventListener('click', () => {
            filterPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');

            const cat = pill.getAttribute('data-filter');
            dishCards.forEach(card => {
                const cardCat = card.getAttribute('data-category');
                if (cat === 'all' || cardCat === cat) {
                    card.style.display = 'flex';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
});
