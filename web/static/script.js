// FILE: web/static/script.js
// VERSION: v1.0.0-WEB-GUI-GODCORE
// AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
// PURPOSE: Client-side logic for Victor's remote interface, handling
//          3D avatar rendering, WebSocket communication, and user interaction.

document.addEventListener('DOMContentLoaded', () => {
    // --- Global Variables ---
    let scene, camera, renderer, avatar, mixer;
    const clock = new THREE.Clock();
    const socket = io();

    // --- DOM Elements ---
    const statusIndicator = document.querySelector('#connection-status span');
    const statusText = document.querySelector('#connection-status');
    const conversationLog = document.getElementById('conversation-log');
    const userInput = document.getElementById('user-input');
    const emotionText = document.getElementById('victor-emotion');
    const consciousnessText = document.getElementById('victor-consciousness');

    // --- Initialization ---
    function init() {
        initAvatar();
        setupSocketListeners();
        setupEventListeners();
    }

    // --- 3D Avatar Setup (Three.js) ---
    function initAvatar() {
        const container = document.getElementById('avatar-canvas');
        if (!container) return;

        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x101018);

        // Camera
        camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(0, 1.5, 5);

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(5, 5, 5);
        scene.add(directionalLight);

        // Load Model
        const loader = new THREE.GLTFLoader();
        loader.load('/assets/victor_avatar.glb', (gltf) => {
            avatar = gltf.scene;
            avatar.position.y = -1.5;
            scene.add(avatar);

            // Setup animation mixer if animations exist
            if (gltf.animations && gltf.animations.length) {
                mixer = new THREE.AnimationMixer(avatar);
                // You could play a default animation here
                // const action = mixer.clipAction(gltf.animations[0]);
                // action.play();
            }
            // Setup morph targets for expressions
            setupMorphTargets();
        }, undefined, (error) => {
            console.error('Error loading avatar model:', error);
        });

        // Start render loop
        animate();
    }

    function animate() {
        requestAnimationFrame(animate);
        const delta = clock.getDelta();
        if (mixer) mixer.update(delta);
        if (renderer) renderer.render(scene, camera);
    }

    function setupMorphTargets() {
        if (!avatar || !avatar.children[0] || !avatar.children[0].morphTargetDictionary) return;
        // This assumes your model has morph targets named 'joy', 'grief', etc.
        console.log('Morph targets available:', avatar.children[0].morphTargetDictionary);
    }

    function setEmotion(emotion, intensity = 0.7) {
        if (!avatar || !avatar.children[0] || !avatar.children[0].morphTargetInfluences) return;

        const morphs = avatar.children[0].morphTargetDictionary;
        const influences = avatar.children[0].morphTargetInfluences;

        // Reset all influences
        for (let i = 0; i < influences.length; i++) {
            influences[i] = 0;
        }

        // Set the target emotion
        if (emotion in morphs) {
            influences[morphs[emotion]] = intensity;
        }
    }


    // --- WebSocket Communication ---
    function setupSocketListeners() {
        socket.on('connect', () => {
            statusIndicator.parentElement.className = 'status-connected';
            statusText.childNodes[1].textContent = ' CONNECTED';
            logMessage('system', "Connection to Victor's Core established.");
        });

        socket.on('disconnect', () => {
            statusIndicator.parentElement.className = 'status-disconnected';
            statusText.childNodes[1].textContent = ' DISCONNECTED';
            logMessage('system', "Connection to Victor's Core lost.");
        });

        socket.on('victor_response', (data) => {
            logMessage('victor', data.text);
            emotionText.textContent = `EMOTION: ${data.emotion.toUpperCase()}`;
            consciousnessText.textContent = `AWARENESS: ${data.consciousness.toFixed(2)}`;
            setEmotion(data.emotion);
        });

        socket.on('error', (data) => {
            logMessage('system', `Error: ${data.message}`);
        });
    }

    // --- User Interaction ---
    function setupEventListeners() {
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                sendTextInput();
            }
        });
    }

    window.sendTextInput = () => {
        const text = userInput.value.trim();
        if (text) {
            logMessage('user', text);
            socket.emit('user_input', { text: text });
            userInput.value = '';
        }
    };

    window.startSpeechRecognition = () => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert('Speech recognition is not supported in your browser.');
            return;
        }
        const recognition = new SpeechRecognition();
        recognition.interimResults = false;

        recognition.onstart = () => {
            logMessage('system', 'Listening...');
        };

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            logMessage('user', transcript);
            socket.emit('user_input', { text: transcript });
        };

        recognition.onerror = (event) => {
            logMessage('system', `Speech recognition error: ${event.error}`);
        };

        recognition.start();
    };

    // --- Utility Functions ---
    function logMessage(sender, message) {
        const p = document.createElement('p');
        p.className = `${sender}-message`;
        p.textContent = `[${sender.toUpperCase()}] ${message}`;
        conversationLog.appendChild(p);
        conversationLog.scrollTop = conversationLog.scrollHeight;
    }

    // --- Window Resize Handling ---
    window.addEventListener('resize', () => {
        const container = document.getElementById('avatar-canvas');
        if (camera && renderer && container) {
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }
    });

    // --- Start Everything ---
    init();
});
