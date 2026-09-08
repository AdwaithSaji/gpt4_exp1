let previousProduct = null;


// ==============================
// GET ELEMENTS
// ==============================

const chatBox = document.getElementById("chat-box");

const userInput = document.getElementById("user-input");

const sendButton = document.getElementById("send-button");

const typing = document.getElementById("typing");


// ==============================
// ADD MESSAGE
// ==============================

function addMessage(message, sender, intent = null) {

    const messageDiv = document.createElement("div");

    if (sender === "user") {

        messageDiv.className = "message user-message";

        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="sender">You</div>
                <div class="text">${formatMessage(message)}</div>
            </div>

            <div class="avatar">👤</div>
        `;

    } else {

        messageDiv.className = "message bot-message";

        let intentHTML = "";

        if (intent) {

            intentHTML = `
                <div class="intent">
                    Detected Intent: ${intent}
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="avatar">🤖</div>

            <div class="message-content">

                <div class="sender">
                    AI Assistant
                </div>

                ${intentHTML}

                <div class="text">
                    ${formatMessage(message)}
                </div>

            </div>
        `;
    }

    chatBox.appendChild(messageDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
}


// ==============================
// FORMAT MESSAGE
// ==============================

function escapeHtml(text) {

    const div = document.createElement("div");

    div.textContent = text;

    return div.innerHTML;
}


function formatMessage(message) {

    if (!message) {
        return "";
    }

    // Escape first, so that anything typed by the user or returned
    // by the model is shown as text instead of being parsed as HTML.
    return escapeHtml(String(message))
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
}


// ==============================
// SEND MESSAGE
// ==============================

async function sendMessage() {

    const message = userInput.value.trim();

    if (!message) {
        return;
    }


    // Show user message
    addMessage(
        message,
        "user"
    );


    // Clear input
    userInput.value = "";


    // Disable button
    sendButton.disabled = true;

    typing.classList.remove("hidden");


    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                message: message,

                previous_product: previousProduct

            })

        });


        const data = await response.json();


        typing.classList.add("hidden");

        sendButton.disabled = false;


        if (data.success) {

            // Remember selected product
            if (data.previous_product) {

                previousProduct = data.previous_product;

            }


            addMessage(
                data.response,
                "bot",
                data.intent
            );

        } else {

            addMessage(
                data.message || "Something went wrong.",
                "bot"
            );

        }

    } catch (error) {

        typing.classList.add("hidden");

        sendButton.disabled = false;

        addMessage(
            "Unable to connect to the server. Please make sure the Flask application is running.",
            "bot"
        );

        console.error(error);
    }

}


// ==============================
// SEND BUTTON
// ==============================

sendButton.addEventListener(
    "click",
    sendMessage
);


// ==============================
// ENTER KEY
// ==============================

userInput.addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {

            sendMessage();

        }

    }
);