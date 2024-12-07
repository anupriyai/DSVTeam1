# DSV Team 1: LLM Evaluation Platform for Data Science Varsity

This repository provides a platform for evaluating large language models (LLMs).

**Getting Started**

Follow the steps below to set up and run the frontend application:

---

**1. Navigate to the Frontend Directory**

Open your terminal and navigate to the `frontend` folder within your project. For example:

```bash
C:\Users\anupr\OneDrive\Desktop\DSV_LLM> cd DSVTeam1
C:\Users\anupr\OneDrive\Desktop\DSV_LLM\DSVTeam1> cd frontend
C:\Users\anupr\OneDrive\Desktop\DSV_LLM\DSVTeam1\frontend>
```

---

**2. Install Dependencies**

### a) Install Node.js (if not already installed)
Download and install Node.js from the [official Node.js website](https://nodejs.org/en/download/prebuilt-installer).

### b) Install Required Packages
Once inside the `frontend` folder, run the following command to install the required dependencies:

```bash
npm install
```

---

**3. Launch the Development Server**

To start the development server, ensure you are still in the `frontend` folder and run:

```bash
npm run dev
```

---

The frontend should now be up and running. You can access it in your browser at the URL provided in the terminal output (typically `http://localhost:3000`).

--- 


**Getting Started**

1.  **Clone the Repository:**

      - Open a terminal (e.g., VS Code terminal).

      - Clone the repository using Git:

        ```bash
        git clone https://github.com/anupriyai/DSVTeam1.git
        ```

2.  **Stay Updated:**

      - To avoid merge conflicts, keep your local copy up-to-date before making changes:

        ```bash
        git pull origin main
        ```

3.  **Create Your Branch:**

      - Create a personal branch to isolate your work and keep the `main` branch clean:

        ```bash
        git branch <your_name>  # Replace `<your_name>` with your actual name
        git checkout <your_name>
        ```

4.  **Setup:**

      - **Create a virtual environment:**

          - This helps manage project dependencies and avoids conflicts with your system's Python environment.

            ```bash
            python3 -m venv venv  # Create a virtual environment named `venv`
            ```

          - Activate the virtual environment:

            ```bash
            source venv/bin/activate  # Linux/macOS
            .\venv\Scripts\activate  # Windows
            ```

      - **Install dependencies:**

        ```bash
        pip install -r requirements.txt
        ```

      - **Create a `.env` file:**

          - This file securely stores the OpenAI API key.

          - Create a file named `.env` in the project's root directory with the following line (replace `<YOUR_OPENAI_KEY>` with the one I provided):

            ```
            OPENAI_API_KEY=<YOUR_OPENAI_KEY>
            ```


5.  **Run the Tests:**

      - **Before committing changes:**

          - Run the tests to ensure your modifications don't break existing functionality:

            ```bash
            deepeval test run test_example.py
            ```

              - This command executes the `test_example.py` script, which demonstrates basic `deepeval` usage.
              - Review the output to verify successful test execution.

      - **After making changes:**

          - After implementing your changes, follow the Git workflow to commit and push your code:

            1.  Add modified files:

                ```bash
                git add <filename.py>  # Replace `<filename.py>` with the actual file
                ```

            2.  Commit your changes with a descriptive message:

                ```bash
                git commit -m "Describe your changes here"
                ```

            3.  Push your branch to GitHub:

                ```bash
                git push origin <your_name>  # Replace `<your_name>` with your branch name
                ```
