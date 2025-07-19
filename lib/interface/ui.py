import streamlit as st
import json
import sys
import os
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from lib.layout.layout import MangaLayout, Speaker, NonSpeaker

def create_drag_interface():
    # セッション状態の初期化
    if 'page_state' not in st.session_state:
        st.session_state.page_state = 'main'
    if 'current_layout' not in st.session_state:
        st.session_state.current_layout = None
    if 'similarity_results' not in st.session_state:
        st.session_state.similarity_results = None
    
    st.title("Manga Layout Editor")
    
    # サイドバーで設定
    st.sidebar.header("設定")
    
    # unrelated_text_lengthのみ設定
    unrelated_text_length = st.sidebar.number_input("Unrelated Text Length", min_value=0, value=100)
    
    # 閾値設定
    aspect_ratio_threshold = st.sidebar.number_input("Aspect Ratio Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    text_length_threshold = st.sidebar.number_input("Text Length Threshold", min_value=0, value=20)
    
    # キャンバスサイズを固定
    canvas_width = 512
    canvas_height = 512
    
    # ページ状態に応じて表示を切り替え
    if st.session_state.page_state == 'main':
        show_main_page(canvas_width, canvas_height, unrelated_text_length, 
                      aspect_ratio_threshold, text_length_threshold)
    elif st.session_state.page_state == 'layout_created':
        show_layout_page(unrelated_text_length, aspect_ratio_threshold, text_length_threshold)
    elif st.session_state.page_state == 'similarity_calculated':
        show_similarity_page()

def show_main_page(canvas_width, canvas_height, unrelated_text_length, 
                  aspect_ratio_threshold, text_length_threshold):
    """メインページ（キャンバスとレイアウト作成ボタン）を表示"""
    
    # HTMLとJavaScriptでドラッグ機能を実装
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            #canvas {{
                border: 2px solid #333;
                cursor: crosshair;
                position: relative;
                background: white;
            }}
            .bbox {{
                position: absolute;
                border: 2px solid;
                background: rgba(255, 0, 0, 0.1);
                cursor: move;
            }}
            .speaker {{
                border-color: red;
            }}
            .nonspeaker {{
                border-color: blue;
            }}
            .bbox.selected {{
                border-width: 4px;
                background: rgba(255, 255, 0, 0.3);
            }}
            .bbox.dragging {{
                opacity: 0.7;
                z-index: 1000;
            }}
            .bbox-label {{
                position: absolute;
                background: white;
                padding: 2px 6px;
                font-size: 12px;
                font-weight: bold;
                pointer-events: none;
            }}
            .edit-panel {{
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ccc;
                border-radius: 5px;
                background: #f9f9f9;
            }}
            .edit-panel input, .edit-panel select {{
                margin: 5px;
                padding: 5px;
            }}
            .edit-panel button {{
                margin: 5px;
                padding: 8px 15px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }}
            .edit-panel button:hover {{
                background: #0056b3;
            }}
            .edit-panel button.delete {{
                background: #dc3545;
            }}
            .edit-panel button.delete:hover {{
                background: #c82333;
            }}
            .mode-indicator {{
                margin: 10px 0;
                padding: 10px;
                background: #e9ecef;
                border-radius: 5px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div style="margin: 20px;">
            <h3>ドラッグしてバウンディングボックスを作成・選択・移動</h3>
            <div class="mode-indicator" id="mode-indicator">モード: 描画</div>
            <div id="canvas" style="width: {canvas_width}px; height: {canvas_height}px;"></div>
            <br>
            <button onclick="toggleMode()">モード切替 (描画/移動)</button>
            <button onclick="clearAll()">全てクリア</button>
            <button onclick="exportData()">データエクスポート</button>
            <button onclick="selectAll()">全選択</button>
            <button onclick="deselectAll()">選択解除</button>
            <button onclick="createLayout()">レイアウトを作成</button>
            <br><br>
            
            <!-- 編集パネル -->
            <div id="edit-panel" class="edit-panel" style="display: none;">
                <h4>選択された要素の編集</h4>
                <div>
                    <label>要素タイプ:</label>
                    <select id="edit-type">
                        <option value="Speaker">Speaker</option>
                        <option value="Non-Speaker">Non-Speaker</option>
                    </select>
                </div>
                <div>
                    <label>文字数:</label>
                    <input type="number" id="edit-text-length" min="0" value="10">
                </div>
                <button onclick="updateSelectedBbox()">更新</button>
                <button onclick="deleteSelectedBbox()" class="delete">削除</button>
                <button onclick="closeEditPanel()">閉じる</button>
            </div>
            
            <div id="bbox-list">
                <h4>作成されたバウンディングボックス:</h4>
                <ul id="bbox-items"></ul>
            </div>
        </div>

        <script>
            let bboxes = [];
            let isDrawing = false;
            let isDragging = false;
            let isMoveMode = false;
            let startX, startY;
            let currentBbox = null;
            let bboxCounter = 0;
            let selectedBboxId = null;
            let dragOffsetX = 0;
            let dragOffsetY = 0;
            let currentElementType = 'Speaker';
            let currentTextLength = 10;

            const canvas = document.getElementById('canvas');
            const bboxItems = document.getElementById('bbox-items');
            const editPanel = document.getElementById('edit-panel');
            const modeIndicator = document.getElementById('mode-indicator');

            canvas.addEventListener('mousedown', startInteraction);
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseup', endInteraction);

            function toggleMode() {{
                isMoveMode = !isMoveMode;
                updateModeIndicator();
                if (isMoveMode) {{
                    canvas.style.cursor = 'move';
                }} else {{
                    canvas.style.cursor = 'crosshair';
                }}
            }}

            function updateModeIndicator() {{
                modeIndicator.textContent = 'モード: ' + (isMoveMode ? '移動' : '描画');
                modeIndicator.style.background = isMoveMode ? '#fff3cd' : '#e9ecef';
            }}

            function startInteraction(e) {{
                console.log('startInteraction called');
                const rect = canvas.getBoundingClientRect();
                startX = e.clientX - rect.left;
                startY = e.clientY - rect.top;
                console.log('Start position:', startX, startY);

                if (isMoveMode) {{
                    console.log('Move mode - checking for bbox');
                    // 移動モード: 既存のbboxをドラッグ
                    const clickedBbox = e.target.closest('.bbox');
                    if (clickedBbox) {{
                        console.log('Bbox clicked for moving');
                        const bboxId = parseInt(clickedBbox.getAttribute('data-id'));
                        selectBbox(bboxId);
                        
                        // ドラッグ開始
                        isDragging = true;
                        const bboxRect = clickedBbox.getBoundingClientRect();
                        dragOffsetX = startX - (bboxRect.left - rect.left);
                        dragOffsetY = startY - (bboxRect.top - rect.top);
                        
                        clickedBbox.classList.add('dragging');
                        e.preventDefault();
                    }}
                }} else {{
                    console.log('Draw mode - creating new bbox');
                    // 描画モード: 新しいbboxを作成
                    const clickedBbox = e.target.closest('.bbox');
                    if (clickedBbox) {{
                        console.log('Existing bbox clicked for selection');
                        const bboxId = parseInt(clickedBbox.getAttribute('data-id'));
                        selectBbox(bboxId);
                        return;
                    }}
                    
                    console.log('Starting to draw new bbox');
                    isDrawing = true;
                    currentBbox = document.createElement('div');
                    currentBbox.className = 'bbox ' + (currentElementType.toLowerCase() === 'speaker' ? 'speaker' : 'nonspeaker');
                    currentBbox.style.left = startX + 'px';
                    currentBbox.style.top = startY + 'px';
                    currentBbox.style.width = '0px';
                    currentBbox.style.height = '0px';
                    canvas.appendChild(currentBbox);
                    console.log('New bbox element created');
                }}
            }}

            function handleMouseMove(e) {{
                if (isDragging || isDrawing) {{
                    console.log('handleMouseMove - dragging:', isDragging, 'drawing:', isDrawing);
                }}
                
                const rect = canvas.getBoundingClientRect();
                const currentX = e.clientX - rect.left;
                const currentY = e.clientY - rect.top;

                if (isDragging && selectedBboxId) {{
                    // 移動処理
                    const bboxElement = canvas.querySelector(`[data-id="${{selectedBboxId}}"]`);
                    if (bboxElement) {{
                        const newX = currentX - dragOffsetX;
                        const newY = currentY - dragOffsetY;
                        
                        // キャンバス内に制限
                        const bboxWidth = bboxElement.offsetWidth;
                        const bboxHeight = bboxElement.offsetHeight;
                        const clampedX = Math.max(0, Math.min(newX, {canvas_width} - bboxWidth));
                        const clampedY = Math.max(0, Math.min(newY, {canvas_height} - bboxHeight));
                        
                        bboxElement.style.left = clampedX + 'px';
                        bboxElement.style.top = clampedY + 'px';
                        
                        // ラベルも移動
                        const label = bboxElement.querySelector('.bbox-label');
                        if (label) {{
                            label.style.left = (bboxWidth / 2 - 10) + 'px';
                            label.style.top = (bboxHeight / 2 - 10) + 'px';
                        }}
                    }}
                }} else if (isDrawing && currentBbox) {{
                    // 描画処理
                    const width = Math.abs(currentX - startX);
                    const height = Math.abs(currentY - startY);
                    const left = Math.min(startX, currentX);
                    const top = Math.min(startY, currentY);
                    
                    currentBbox.style.left = left + 'px';
                    currentBbox.style.top = top + 'px';
                    currentBbox.style.width = width + 'px';
                    currentBbox.style.height = height + 'px';
                }}
            }}

            function endInteraction(e) {{
                console.log('endInteraction called - dragging:', isDragging, 'drawing:', isDrawing);
                
                if (isDragging) {{
                    // 移動終了
                    isDragging = false;
                    const bboxElement = canvas.querySelector(`[data-id="${{selectedBboxId}}"]`);
                    if (bboxElement) {{
                        bboxElement.classList.remove('dragging');
                        
                        // データを更新
                        const bboxData = bboxes.find(b => b.id === selectedBboxId);
                        if (bboxData) {{
                            const rect = bboxElement.getBoundingClientRect();
                            const canvasRect = canvas.getBoundingClientRect();
                            const x1 = rect.left - canvasRect.left;
                            const y1 = rect.top - canvasRect.top;
                            const x2 = x1 + rect.width;
                            const y2 = y1 + rect.height;
                            
                            bboxData.bbox = [x1, y1, x2, y2];
                            updateBboxList();
                        }}
                    }}
                }} else if (isDrawing && currentBbox) {{
                    // 描画終了
                    console.log('Drawing finished');
                    const rect = canvas.getBoundingClientRect();
                    const endX = e.clientX - rect.left;
                    const endY = e.clientY - rect.top;
                    
                    const x1 = Math.min(startX, endX);
                    const y1 = Math.min(startY, endY);
                    const x2 = Math.max(startX, endX);
                    const y2 = Math.max(startY, endY);
                    
                    console.log('Bbox dimensions:', x1, y1, x2, y2);
                    
                    // 最小サイズチェック
                    if (x2 - x1 > 10 && y2 - y1 > 10) {{
                        console.log('Creating bbox data');
                        bboxCounter++;
                        
                        // データを保存
                        const bboxData = {{
                            id: bboxCounter,
                            type: currentElementType,
                            bbox: [x1, y1, x2, y2],
                            text_length: currentTextLength
                        }};
                        bboxes.push(bboxData);
                        console.log('Bbox added to array, total:', bboxes.length);
                        
                        // DOM要素を更新
                        updateBboxElement(currentBbox, bboxData);
                        
                        // リストに追加
                        updateBboxList();
                    }} else {{
                        console.log('Bbox too small, removing');
                        canvas.removeChild(currentBbox);
                    }}
                    
                    isDrawing = false;
                    currentBbox = null;
                }}
            }}

            function updateBboxElement(element, bboxData) {{
                element.setAttribute('data-id', bboxData.id);
                element.className = 'bbox ' + (bboxData.type.toLowerCase() === 'speaker' ? 'speaker' : 'nonspeaker');
                
                // ラベルを追加
                const label = document.createElement('div');
                label.className = 'bbox-label';
                label.textContent = bboxData.id;
                label.style.left = (bboxData.bbox[0] + (bboxData.bbox[2] - bboxData.bbox[0]) / 2 - 10) + 'px';
                label.style.top = (bboxData.bbox[1] + (bboxData.bbox[3] - bboxData.bbox[1]) / 2 - 10) + 'px';
                element.appendChild(label);
            }}

            function selectBbox(bboxId) {{
                // 既存の選択を解除
                deselectAll();
                
                // 新しい選択
                selectedBboxId = bboxId;
                const bboxElement = canvas.querySelector(`[data-id="${{bboxId}}"]`);
                if (bboxElement) {{
                    bboxElement.classList.add('selected');
                }}
                
                // 編集パネルを表示
                showEditPanel(bboxId);
            }}

            function deselectAll() {{
                const selectedElements = canvas.querySelectorAll('.bbox.selected');
                selectedElements.forEach(el => el.classList.remove('selected'));
                selectedBboxId = null;
                editPanel.style.display = 'none';
            }}

            function selectAll() {{
                const bboxElements = canvas.querySelectorAll('.bbox');
                bboxElements.forEach(el => el.classList.add('selected'));
            }}

            function showEditPanel(bboxId) {{
                const bboxData = bboxes.find(b => b.id === bboxId);
                if (bboxData) {{
                    document.getElementById('edit-type').value = bboxData.type;
                    document.getElementById('edit-text-length').value = bboxData.text_length || 0;
                    editPanel.style.display = 'block';
                }}
            }}

            function updateSelectedBbox() {{
                if (!selectedBboxId) return;
                
                const bboxData = bboxes.find(b => b.id === selectedBboxId);
                if (bboxData) {{
                    bboxData.type = document.getElementById('edit-type').value;
                    bboxData.text_length = parseInt(document.getElementById('edit-text-length').value) || 0;
                    
                    // 現在の設定を更新（次回の描画用）
                    currentElementType = bboxData.type;
                    currentTextLength = bboxData.text_length;
                    
                    // DOM要素を更新
                    const bboxElement = canvas.querySelector(`[data-id="${{selectedBboxId}}"]`);
                    if (bboxElement) {{
                        bboxElement.className = 'bbox ' + (bboxData.type.toLowerCase() === 'speaker' ? 'speaker' : 'nonspeaker');
                    }}
                    
                    updateBboxList();
                }}
            }}

            function deleteSelectedBbox() {{
                if (!selectedBboxId) return;
                
                // データから削除
                bboxes = bboxes.filter(b => b.id !== selectedBboxId);
                
                // DOM要素から削除
                const bboxElement = canvas.querySelector(`[data-id="${{selectedBboxId}}"]`);
                if (bboxElement) {{
                    canvas.removeChild(bboxElement);
                }}
                
                selectedBboxId = null;
                editPanel.style.display = 'none';
                updateBboxList();
            }}

            function closeEditPanel() {{
                editPanel.style.display = 'none';
                deselectAll();
            }}

            function updateBboxList() {{
                bboxItems.innerHTML = '';
                bboxes.forEach(bbox => {{
                    const li = document.createElement('li');
                    li.textContent = `${{bbox.id}}. ${{bbox.type}} - [${{bbox.bbox.join(', ')}}]`;
                    if (bbox.type === 'Speaker') {{
                        li.textContent += ` (文字数: ${{bbox.text_length}})`;
                    }}
                    
                    // クリックで選択
                    li.style.cursor = 'pointer';
                    li.onclick = () => selectBbox(bbox.id);
                    if (bbox.id === selectedBboxId) {{
                        li.style.fontWeight = 'bold';
                        li.style.color = '#007bff';
                    }}
                    
                    bboxItems.appendChild(li);
                }});
            }}

            function clearAll() {{
                bboxes = [];
                bboxCounter = 0;
                selectedBboxId = null;
                const bboxElements = canvas.querySelectorAll('.bbox');
                bboxElements.forEach(el => el.remove());
                editPanel.style.display = 'none';
                updateBboxList();
            }}

            function exportData() {{
                const data = {{
                    canvas_width: {canvas_width},
                    canvas_height: {canvas_height},
                    unrelated_text_length: {unrelated_text_length},
                    bboxes: bboxes
                }};
                
                // Streamlitにデータを送信
                if (window.parent && window.parent.postMessage) {{
                    window.parent.postMessage({{
                        type: 'export_data',
                        data: data
                    }}, '*');
                }}
                
                console.log('Exported data:', data);
                alert('データをエクスポートしました！');
            }}

            function createLayout() {{
                console.log('createLayout called');
                console.log('getBboxData called, bboxes:', bboxes);
                console.log('bboxes length:', bboxes.length);
                console.log('bboxes type:', typeof bboxes);
                
                if (bboxes && bboxes.length > 0) {{
                    console.log('Data found:', bboxes);
                    
                    // データをtemp.jsonとしてローカルに保存
                    const dataToSave = {{
                        canvas_width: {canvas_width},
                        canvas_height: {canvas_height},
                        unrelated_text_length: {unrelated_text_length},
                        bboxes: bboxes
                    }};
                    
                    // データをJSONファイルとしてダウンロード
                    const dataStr = JSON.stringify(dataToSave, null, 2);
                    const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
                    const url = URL.createObjectURL(dataBlob);
                    
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'temp.json';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(url);
                    
                    console.log('Data saved to temp.json');
                    alert('データがtemp.jsonとして保存されました。ファイルをアップロードしてレイアウトを作成してください。');
                }} else {{
                    console.log('No data found');
                    alert('バウンディングボックスが作成されていません。キャンバス上でバウンディングボックスを作成してください。');
                }}
            }}

            // データを取得する関数をグローバルに公開
            window.getBboxData = function() {{
                console.log('getBboxData called, bboxes:', bboxes);
                console.log('bboxes length:', bboxes.length);
                console.log('bboxes type:', typeof bboxes);
                return bboxes;
            }};
            
            // URLパラメータからデータを復元
            function restoreFromUrl() {{
                const urlParams = new URLSearchParams(window.location.search);
                const bboxDataParam = urlParams.get('bbox_data');
                if (bboxDataParam) {{
                    try {{
                        const decodedData = decodeURIComponent(bboxDataParam);
                        const restoredBboxes = JSON.parse(decodedData);
                        console.log('Restoring bboxes from URL:', restoredBboxes);
                        
                        bboxes = restoredBboxes;
                        bboxCounter = Math.max(...bboxes.map(b => b.id), 0);
                        
                        // DOM要素を再作成
                        bboxes.forEach(bboxData => {{
                            const bboxElement = document.createElement('div');
                            bboxElement.className = 'bbox ' + (bboxData.type.toLowerCase() === 'speaker' ? 'speaker' : 'nonspeaker');
                            bboxElement.setAttribute('data-id', bboxData.id);
                            
                            const [x1, y1, x2, y2] = bboxData.bbox;
                            bboxElement.style.left = x1 + 'px';
                            bboxElement.style.top = y1 + 'px';
                            bboxElement.style.width = (x2 - x1) + 'px';
                            bboxElement.style.height = (y2 - y1) + 'px';
                            
                            // ラベルを追加
                            const label = document.createElement('div');
                            label.className = 'bbox-label';
                            label.textContent = bboxData.id;
                            label.style.left = ((x2 - x1) / 2 - 10) + 'px';
                            label.style.top = ((y2 - y1) / 2 - 10) + 'px';
                            bboxElement.appendChild(label);
                            
                            canvas.appendChild(bboxElement);
                        }});
                        
                        updateBboxList();
                        console.log('Bboxes restored successfully');
                    }} catch (error) {{
                        console.error('Error restoring bboxes:', error);
                    }}
                }}
            }}

            // 初期化時にデバッグ情報を表示
            console.log('JavaScript initialized, bboxes:', bboxes);
            console.log('getBboxData function:', window.getBboxData);

            // 初期化
            updateModeIndicator();
            
            // URLからデータを復元
            restoreFromUrl();
        </script>
    </body>
    </html>
    """
    
    # HTMLを表示
    st.components.v1.html(html_code, height=canvas_height + 400)
    
    # ファイルアップロード機能
    st.subheader("レイアウトデータのアップロード")
    uploaded_file = st.file_uploader("temp.jsonファイルをアップロード", type=['json'])
    
    if uploaded_file is not None:
        try:
            # JSONファイルを読み込み
            data = json.load(uploaded_file)
            st.write(f"アップロードされたデータ: {data}")
            
            # データ形式を変換
            converted_data = []
            for bbox in data['bboxes']:
                x1, y1, x2, y2 = bbox['bbox']
                converted_data.append({
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'type': bbox['type'].lower(),
                    'textLength': bbox.get('text_length', 0)
                })
            
            st.write(f"変換後のデータ: {converted_data}")
            
            # レイアウトを作成
            layout = create_manga_layout_object(converted_data, data['canvas_width'], data['canvas_height'], 
                                              data['unrelated_text_length'], aspect_ratio_threshold, text_length_threshold)
            
            # セッション状態を更新
            st.session_state.current_layout = layout
            st.session_state.page_state = 'layout_created'
            st.rerun()
            
        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {e}")
            st.exception(e)
    

    


def show_layout_page(unrelated_text_length, aspect_ratio_threshold, text_length_threshold):
    """レイアウト作成後のページを表示"""
    st.subheader("作成されたレイアウト")
    
    layout = st.session_state.current_layout
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**キャンバスサイズ**: {layout.width} × {layout.height}")
        st.write(f"**要素数**: {len(layout.elements)}")
        st.write(f"**Speaker要素**: {sum(1 for e in layout.elements if isinstance(e, Speaker))}")
        st.write(f"**NonSpeaker要素**: {sum(1 for e in layout.elements if isinstance(e, NonSpeaker))}")
        st.write(f"**Unrelated Text Length**: {unrelated_text_length}")
    
    with col2:
        st.write(f"**Aspect Ratio Threshold**: {aspect_ratio_threshold}")
        st.write(f"**Text Length Threshold**: {text_length_threshold}")
        st.write(f"**総文字数**: {sum(getattr(e, 'text_length', 0) for e in layout.elements if isinstance(e, Speaker))}")
    
    # 要素の詳細情報
    st.subheader("要素詳細")
    for i, element in enumerate(layout.elements):
        element_type = "Speaker" if isinstance(element, Speaker) else "NonSpeaker"
        text_info = f" (文字数: {element.text_length})" if isinstance(element, Speaker) else ""
        st.write(f"**要素 {i+1}**: {element_type}{text_info} - BBox: {element.bbox}")
    
    # 類似度計算ボタン
    if st.button("類似度計算を実行", key="similarity_calc"):
        with st.spinner("類似度計算中..."):
            run_similarity_calculation(layout, aspect_ratio_threshold, text_length_threshold)
            st.session_state.page_state = 'similarity_calculated'
            st.rerun()
    
    # メインページに戻るボタン
    if st.button("新しいレイアウトを作成"):
        st.session_state.page_state = 'main'
        st.session_state.current_layout = None
        st.session_state.similarity_results = None
        st.rerun()

def show_similarity_page():
    """類似度計算結果ページを表示"""
    st.subheader("類似度計算結果")
    
    if st.session_state.similarity_results:
        # 結果表示
        st.write("**上位5つの類似レイアウト:**")
        for i, (other_layout, score) in enumerate(st.session_state.similarity_results):
            st.write(f"**Rank {i+1}**: Score = {score:.4f}")
            if hasattr(other_layout, 'image_path') and other_layout.image_path:
                st.write(f"  - Image: {other_layout.image_path}")
            st.write(f"  - Elements: {len(other_layout.elements)}")
        
        # Gallery表示ボタン
        if st.button("Gallery表示", key="show_gallery"):
            display_similarity_gallery(st.session_state.current_layout, st.session_state.similarity_results)
    
    # メインページに戻るボタン
    if st.button("新しいレイアウトを作成", key="new_layout"):
        st.session_state.page_state = 'main'
        st.session_state.current_layout = None
        st.session_state.similarity_results = None
        st.rerun()

def create_manga_layout_object(bbox_data, canvas_width, canvas_height, unrelated_text_length, 
                              aspect_ratio_threshold, text_length_threshold):
    """バウンディングボックスデータからMangaLayoutオブジェクトを作成"""
    
    # 要素を作成
    elements = []
    for bbox in bbox_data:
        x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        if bbox['type'] == 'speaker':
            element = Speaker(
                bbox=[x, y, x + width, y + height],
                text_length=bbox.get('textLength', 0)
            )
        else:
            element = NonSpeaker(
                bbox=[x, y, x + width, y + height]
            )
        elements.append(element)
    
    # MangaLayoutを作成
    layout = MangaLayout(
        image_path="",
        width=canvas_width,
        height=canvas_height,
        elements=elements,
        unrelated_text_length=unrelated_text_length
    )
    
    return layout

def create_manga_layout(bbox_data, canvas_width, canvas_height, unrelated_text_length, 
                       aspect_ratio_threshold, text_length_threshold):
    """バウンディングボックスデータからMangaLayoutを作成"""
    
    # 要素を作成
    elements = []
    for bbox in bbox_data:
        x, y, width, height = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        if bbox['type'] == 'speaker':
            element = Speaker(
                bbox=[x, y, x + width, y + height],
                text_length=bbox.get('textLength', 0)
            )
        else:
            element = NonSpeaker(
                bbox=[x, y, x + width, y + height]
            )
        elements.append(element)
    
    # MangaLayoutを作成
    layout = MangaLayout(
        image_path="",
        width=canvas_width,
        height=canvas_height,
        elements=elements,
        unrelated_text_length=unrelated_text_length
    )
    
    # レイアウト情報を表示
    st.subheader("作成されたレイアウト")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**キャンバスサイズ**: {canvas_width} × {canvas_height}")
        st.write(f"**要素数**: {len(elements)}")
        st.write(f"**Speaker要素**: {sum(1 for e in elements if isinstance(e, Speaker))}")
        st.write(f"**NonSpeaker要素**: {sum(1 for e in elements if isinstance(e, NonSpeaker))}")
        st.write(f"**Unrelated Text Length**: {unrelated_text_length}")
    
    with col2:
        st.write(f"**Aspect Ratio Threshold**: {aspect_ratio_threshold}")
        st.write(f"**Text Length Threshold**: {text_length_threshold}")
        st.write(f"**総文字数**: {sum(getattr(e, 'text_length', 0) for e in elements if isinstance(e, Speaker))}")
    
    # 要素の詳細情報
    st.subheader("要素詳細")
    for i, element in enumerate(elements):
        element_type = "Speaker" if isinstance(element, Speaker) else "NonSpeaker"
        text_info = f" (文字数: {element.text_length})" if isinstance(element, Speaker) else ""
        st.write(f"**要素 {i+1}**: {element_type}{text_info} - BBox: {element.bbox}")
    
    # レイアウトをセッション状態に保存
    st.session_state.current_layout = layout
    
    # 類似度計算ボタン
    if st.button("類似度計算を実行", key="similarity_calc"):
        with st.spinner("類似度計算中..."):
            run_similarity_calculation(layout, aspect_ratio_threshold, text_length_threshold)
    
    st.success("レイアウトが正常に作成されました！")

def run_similarity_calculation(layout, aspect_ratio_threshold, text_length_threshold):
    """類似度計算を実行"""
    try:
        from lib.layout.score import calc_similarity
        from lib.layout.layout import from_condition
        import matplotlib.pyplot as plt
        
        st.subheader("類似度計算結果")
        
        # 条件を設定
        annfile = "./curated_dataset/database.json"
        num_speakers = sum(1 for elem in layout.elements if isinstance(elem, Speaker))
        num_non_speakers = sum(1 for elem in layout.elements if isinstance(elem, NonSpeaker))
        base_text_length = sum(getattr(elem, 'text_length', 0) for elem in layout.elements if isinstance(elem, Speaker))
        base_width = layout.width
        base_height = layout.height
        
        # レイアウトを検索
        layouts = from_condition(annfile, num_speakers, num_non_speakers, base_text_length, 
                               text_length_threshold, base_width, base_height, aspect_ratio_threshold, True)
        
        if layouts:
            # スコア計算
            layout_scores = []
            for other_layout in layouts:
                score = calc_similarity(layout, other_layout, 0.4)
                layout_scores.append((other_layout, score))
            
            # ソート
            layout_scores.sort(key=lambda x: x[1], reverse=True)
            top_5_layouts = layout_scores[:5]
            
            # 結果をセッション状態に保存
            st.session_state.similarity_results = top_5_layouts
            
            # 結果表示
            st.write("**上位5つの類似レイアウト:**")
            for i, (other_layout, score) in enumerate(top_5_layouts):
                st.write(f"**Rank {i+1}**: Score = {score:.4f}")
                if hasattr(other_layout, 'image_path') and other_layout.image_path:
                    st.write(f"  - Image: {other_layout.image_path}")
                st.write(f"  - Elements: {len(other_layout.elements)}")
            
            st.success(f"類似度計算完了: {len(layouts)}個のレイアウトを比較")
            
            # Gallery表示ボタン
            if st.button("Gallery表示", key="show_gallery"):
                display_similarity_gallery(layout, top_5_layouts)
        else:
            st.warning("条件に合うレイアウトが見つかりませんでした")
            
    except Exception as e:
        st.error(f"類似度計算に失敗しました: {e}")
        st.exception(e)

def display_similarity_gallery(base_layout, top_layouts):
    """類似レイアウトのGallery表示"""
    st.subheader("類似レイアウト Gallery")
    
    # 元のレイアウトと類似レイアウトを並べて表示
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 元のレイアウトを表示
    ax = axes[0, 0]
    ax.set_title("元のレイアウト", fontsize=12)
    base_layout.plot_data(ax)
    
    # 類似レイアウトを表示
    for i, (layout, score) in enumerate(top_layouts):
        row = (i + 1) // 3
        col = (i + 1) % 3
        if row < 2:  # 2行目まで
            ax = axes[row, col]
            ax.set_title(f"Rank {i+1}: Score = {score:.4f}", fontsize=10)
            layout.plot_data(ax)
    
    # 空のサブプロットを非表示
    for i in range(len(top_layouts) + 1, 6):
        row = i // 3
        col = i % 3
        if row < 2:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 詳細情報
    st.subheader("詳細情報")
    for i, (layout, score) in enumerate(top_layouts):
        with st.expander(f"Rank {i+1}: Score = {score:.4f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**画像パス**: {layout.image_path}")
                st.write(f"**サイズ**: {layout.width} × {layout.height}")
                st.write(f"**要素数**: {len(layout.elements)}")
            
            with col2:
                speaker_count = sum(1 for elem in layout.elements if isinstance(elem, Speaker))
                nonspeaker_count = sum(1 for elem in layout.elements if isinstance(elem, NonSpeaker))
                st.write(f"**Speaker**: {speaker_count}")
                st.write(f"**NonSpeaker**: {nonspeaker_count}")
                st.write(f"**Unrelated Text**: {layout.unrelated_text_length}")
            
            # 要素詳細
            st.write("**要素詳細:**")
            for j, element in enumerate(layout.elements):
                element_type = "Speaker" if isinstance(element, Speaker) else "NonSpeaker"
                text_info = f" (文字数: {element.text_length})" if isinstance(element, Speaker) else ""
                st.write(f"  - 要素 {j+1}: {element_type}{text_info} - BBox: {element.bbox}")

def main():
    st.set_page_config(page_title="Manga Layout Editor", layout="wide")
    create_drag_interface()

if __name__ == "__main__":
    main()
