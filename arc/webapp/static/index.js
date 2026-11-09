// The poor man's jQuery
const $ = (q,el=document.body)=>el.querySelector(q);

const LEVEL_SET_NAMES_AND_PATHS = [
    ['ARC-1 Training Set','arc-1-train'],
    ['ARC-1 Evaluation Set','arc-1-eval'],
    ['ARC-2 Training Set','arc-2-train'],
    ['ARC-2 Evaluation Set','arc-2-eval'],
    ['Concept ARC','concept-arc'],
    ['Mini ARC','mini-arc'],
    ['Custom','custom']
];

/////////////////////////////////////////////////////////
// FETCHING Level Lists & individual Levels /////////////
/////////////////////////////////////////////////////////

window.levelSets = {};
window.currentLevelSet = null;
window.levelID = null;
window.levelDatas = {};
window.levelSolutions = {};
window.levelPasses = {};

// Fetch all Level Sets [full name, ID]
const fetchLevelSet = async (name)=>{
    const lines = await fetch(`levels/${name}.txt`) // get file
        .then(r=>r.text()) // get text
        .then(text=>text.trim().split('\n').filter(x=>x.trim()!='')) // clean up
    const labelValuePairs = lines.map(line=>[line, line.split('-')[0].trim()]); // [full level name, ID]
    return labelValuePairs;
}

// Load all level sets asynchronously and wait for completion
const loadAllLevelSets = async () => {
    const promises = LEVEL_SET_NAMES_AND_PATHS.map(async ([name,folderPath]) => {
        window.levelSets[folderPath] = await fetchLevelSet(folderPath);
    });
    await Promise.all(promises);
};
await loadAllLevelSets();

// Fetch one level
const fetchLevelByID = async (id)=>{
    window.levelID = id;
    if(window.levelDatas[id]) return window.levelDatas[id]; // if already cached, return!
    // Wait crap which folder was it in...
    let myFolder = null;
    for(let i=0;i<LEVEL_SET_NAMES_AND_PATHS.length;i++){
        let folderPath = LEVEL_SET_NAMES_AND_PATHS[i][1];
        let levelNameAndIDPairs = window.levelSets[folderPath];
        if( levelNameAndIDPairs.some((pair)=>(id==pair[1])) ){ // level ID is in THIS folder.
            myFolder = folderPath;
            break;
        }
    }
    // Else, get it & cache it
    const levelData = await fetch(`levels/${myFolder}/${id}.json`).then(r=>r.json());
    window.levelDatas[id] = levelData;
    return levelData;
}

/////////////////////////////////////////////////////////
// UI: Level Selector Dropdown Menus ////////////////////
/////////////////////////////////////////////////////////

// Fill out the Level Sets
LEVEL_SET_NAMES_AND_PATHS.forEach(labelValue=>{
    const optionDOM = document.createElement("option");
    optionDOM.innerText = labelValue[0];
    optionDOM.value = labelValue[1];
    $('#level_set').appendChild(optionDOM);
});

// Track which index is selected in the current level list.
// Using index (not level ID) because the same ID can appear multiple times.
window.currentLevelIndex = -1;

// When you choose a level SET, populate the Level Select list
$('#level_set').onchange = ()=>{

    // Current level set
    const key = $('#level_set').value;
    const levelSet = window.levelSets[key];
    window.currentLevelSet = key;

    // Reset the selected index when switching sets
    window.currentLevelIndex = -1;

    // Clear the list
    const list = $('#level_select');
    list.innerHTML = '';

    // Each item (with #)
    let i = 0;
    levelSet.forEach(([name,id])=>{
        i++;
        const item = document.createElement('div');
        item.id = `lvl_${id}`;
        item.innerHTML = `<span id="passes_${id}"></span> ${i}: ${name}`;
        item.onclick = async ()=>{
            // Track this item's index (find it by position in the list)
            const allItems = list.children;
            for(let j = 0; j < allItems.length; j++){
                if(allItems[j] === item){
                    window.currentLevelIndex = j;
                    break;
                }
            }
            SHOW_LEVEL(id);
        }
        list.appendChild(item);
    });

}

// Navigate the level list with up/down arrow keys
document.addEventListener('keydown', (event) => {
    // Don't hijack arrows when typing in an input/select
    if(event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;

    const list = $('#level_select');
    const allItems = list.children;
    if(allItems.length === 0) return;

    if(event.key === 'ArrowDown'){
        event.preventDefault();
        const nextIndex = Math.min(window.currentLevelIndex + 1, allItems.length - 1);
        if(nextIndex !== window.currentLevelIndex){
            window.currentLevelIndex = nextIndex;
            allItems[nextIndex].click();
            // Scroll the selected item into view
            allItems[nextIndex].scrollIntoView({ block: 'nearest' });
        }
    } else if(event.key === 'ArrowUp'){
        event.preventDefault();
        const nextIndex = Math.max(window.currentLevelIndex - 1, 0);
        if(nextIndex !== window.currentLevelIndex){
            window.currentLevelIndex = nextIndex;
            allItems[nextIndex].click();
            // Scroll the selected item into view
            allItems[nextIndex].scrollIntoView({ block: 'nearest' });
        }
    }
});

// Let's load the ARC-1 Training Set by default.
$('#level_set').onchange();

/////////////////////////////////////////////////////////
// UI: Show all demonstration & test pairs //////////////
/////////////////////////////////////////////////////////

const illustrateLevel = (levelID)=>{

    const levelData = window.levelDatas[levelID];
    const levelSolution = window.levelSolutions[levelID];

    // Sidebar:
    let currentlyLit = $(".lit")
    if(currentlyLit) currentlyLit.classList.remove('lit');
    $(`#lvl_${levelID}`).classList.add('lit')

    // All demonstrations
    $("#level_id").innerText = 'level #'+levelID;
    $('#official_link').href = `https://arcprize.org/tasks/${levelID}#model-performance`;
    $('#official_link').style.display = window.currentLevelSet.indexOf('eval')>0 ? 'inline' : 'none';
    const demonstrationsDOM = $('#demonstrations');
    demonstrationsDOM.innerHTML = ''; // clear
    levelData.train.forEach(pair=>{

        // Add the pair
        const pairDOM = document.createElement("div");
        pairDOM.className = 'pair';
        demonstrationsDOM.appendChild(pairDOM);

        // The examples
        const examplesDOM = document.createElement("div");
        examplesDOM.className = 'examples';
        pairDOM.appendChild(examplesDOM);
        examplesDOM.appendChild( illustrateGrid(pair.input) );
        examplesDOM.appendChild( _span('→') );
        examplesDOM.appendChild( illustrateGrid(pair.output) );
        
    });

    // All tests
    const testsDOM = $('#tests');
    testsDOM.innerHTML = ''; // clear
    $("#evolution_cot").innerHTML = ''; // clear
    $('#bayes').innerHTML = ''; // clear
    $('#solve_time').innerText = ''; // clear
    levelData.test.forEach(pair=>{
        
        // Add the pair
        const pairDOM = document.createElement("div");
        pairDOM.className = 'pair';
        testsDOM.appendChild(pairDOM);

        // The examples
        const examplesDOM = document.createElement("div");
        examplesDOM.className = 'examples';
        pairDOM.appendChild(examplesDOM);
        examplesDOM.appendChild( illustrateGrid(pair.input) ); // Input image
        examplesDOM.appendChild( _span('→') );
        const attempt1 = illustrateGrid([[0,0,0],[0,0,0],[0,0,0]]); // Attempt 1
        attempt1.id = "attempt1";
        examplesDOM.appendChild( attempt1 );
        examplesDOM.appendChild( _span('or') );
        const attempt2 = illustrateGrid([[0,0,0],[0,0,0],[0,0,0]]); // Attempt 2
        attempt2.id = "attempt2";
        examplesDOM.appendChild( attempt2 );
        examplesDOM.appendChild( _span('≟') );
        let soln = illustrateGrid(pair.output)
        soln.className = "solution"
        examplesDOM.appendChild(soln); // Output image

    });

    // IF SOLUTION EXISTS, SHOW IT.
    if(levelSolution && !levelSolution.error){

        // Show evolution's CoT
        $('#evolution_cot').innerHTML = _syntaxify(levelSolution.reasoning_trace);

        // And the approximate posterior, if the solver sent one (it doesn't if it found no programs at all)
        if(levelSolution.bayes_posterior) illustrateBayesPosterior(levelSolution.bayes_posterior);

        // Replace the images
        levelSolution.test_solutions.forEach((test_solution,i)=>{
            const attempt1 = illustrateGrid( test_solution.attempt_1 );
            attempt1.id = 'attempt1';
            $(`#tests > .pair:nth-child(${i+1}) #attempt1`).replaceWith(attempt1);
            //$(`#tests > .pair:nth-child(${i+1}) #reasoning1`).innerHTML = _syntaxify(test_solution.attempt_1.reasoning);
            const attempt2 = illustrateGrid( test_solution.attempt_2 );
            attempt2.id = 'attempt2';
            $(`#tests > .pair:nth-child(${i+1}) #attempt2`).replaceWith(attempt2);
            //$(`#tests > .pair:nth-child(${i+1}) #reasoning2`).innerHTML = _syntaxify(test_solution.attempt_2.reasoning);
        });

    }

}
// The approximate Bayesian posterior: two kinds of collapsible sections, both collapsed by default
const illustrateBayesPosterior = (bayesPosterior)=>{

    // Over programs: most probable first, each with its probability bar, syntax-colored
    const programsBody = _bayesSection('Approximated Posterior for Programs');
    bayesPosterior.programs.forEach(program=>{
        const programDOM = document.createElement('div');
        programDOM.className = 'bayes_program';
        programDOM.innerHTML = program.is_dummy
            // The dummy stands out: bold white on black
            ? `<b style="color:#fff; background:#000; padding:2px 6px;">${_escapeHTML(program.program)}</b>`
            : _syntaxify(_escapeHTML(program.program));
        programsBody.appendChild( _bayesRow(program.probability, programDOM) );
    });

    // Over each test's outputs: programs that give the same grid pool their probability.
    // The dummy program has no grid, so it's drawn as the big red X
    bayesPosterior.tests.forEach((outputs,i)=>{
        const testBody = _bayesSection(`Approximated Posterior for Test ${i} Outputs`);
        outputs.forEach(output=>{
            testBody.appendChild( _bayesRow(output.probability, illustrateGrid(output.is_dummy ? RED_CROSS : output.grid)) );
        });
    });

}
// A collapsible section (the browser draws the arrow), added to #bayes. Returns its scrollable body
const _bayesSection = (title)=>{
    const details = document.createElement('details'); // closed until clicked
    const summary = document.createElement('summary');
    summary.innerText = title;
    const body = document.createElement('div');
    body.className = 'bayes_body';
    details.appendChild(summary);
    details.appendChild(body);
    $('#bayes').appendChild(details);
    return body;
}
// One entry: the probability bar first, then whatever it's the probability of
const _bayesRow = (probability, contentDOM)=>{
    const row = document.createElement('div');
    row.className = 'bayes_row';
    row.appendChild( _probabilityBar(probability) );
    row.appendChild( contentDOM );
    return row;
}
// Gray track, white fill as wide as the probability, black text on top
const _probabilityBar = (probability)=>{
    const bar = document.createElement('div');
    bar.className = 'probability_bar';
    const fill = document.createElement('div');
    fill.className = 'probability_fill';
    fill.style.width = (probability*100)+'%';
    const label = document.createElement('span');
    label.className = 'probability_text';
    label.innerText = _formatProbability(probability);
    bar.appendChild(fill);
    bar.appendChild(label);
    return bar;
}
// As a percentage, 2 significant figures ("37%", "4.2%", "0.051%"), and anything under 0.01% as "0.00%"
const _formatProbability = (probability)=>{
    const percent = probability*100;
    if(percent >= 10) return percent.toFixed(0)+'%'; // (toPrecision(2) would write 100 as "1.0e+2")
    if(percent >= 0.01) return percent.toPrecision(2)+'%';
    return percent.toFixed(2)+'%';
}
// Programs can contain "<" (e.g. comparisons), which innerHTML would read as a tag
const _escapeHTML = (text)=>{
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
// The solver's "no answer" grid (same as RED_CROSS in alexnot/main_solver.py): a red X on black
const RED_CROSS = [
    [2,0,0,0,0,0,2],
    [0,2,0,0,0,2,0],
    [0,0,2,0,2,0,0],
    [0,0,0,2,0,0,0],
    [0,0,2,0,2,0,0],
    [0,2,0,0,0,2,0],
    [2,0,0,0,0,0,2]
];

const _span = (label)=>{
    const span = document.createElement("span");
    span.innerText = label;
    return span;
}
const COLORS = [
    '#000000', // black
    '#1E93FF', // blue
    '#ff4040', // red
    '#4FCC30', // green
    '#FFDC00', // yellow
    '#999999', // gray
    '#E53AA3', // pink
    '#FF851B', // orange
    '#87D8F1', // sky
    '#921231'  // maroon
]
const COLORS_ARC_3 = [
    "#FFFFFF", // White
    "#CCCCCC", // Off-white
    "#999999", // Light gray
    "#666666", // Gray
    "#333333", // Off-black
    "#000000", // Black
    "#E53AA3", // Magenta
    "#FF7BCC", // Light magenta
    "#F93C31", // Red
    "#1E93FF", // Blue
    "#88D8F1", // Light blue
    "#FFDC00", // Yellow
    "#FF851B", // Orange
    "#921231", // Maroon
    "#4FCC30", // Green
    "#A356D6" // Purple
]
const CANVAS_WIDTH = 200;
const illustrateGrid = (grid)=>{
    
    // Make canvas & context
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // It's 300x?, preserve aspect ratio.
    const scale = CANVAS_WIDTH*2/grid[0].length;
    canvas.width = CANVAS_WIDTH*2; // for retina
    canvas.height = grid.length * scale;
    canvas.style.width = (canvas.width/2)+'px';
    canvas.style.height = (canvas.height/2)+'px';
    
    // Scale *after* setting the canvas
    ctx.scale(scale,scale);
    
    // Fill in each pixel
    for(let x=0; x<grid[0].length; x++){
        for(let y=0; y<grid.length; y++){
            const pixel = grid[y][x];
            const color = grid.length===64 ? COLORS_ARC_3[pixel] : COLORS[pixel]; // ARC-3 uses diff pallette
            ctx.fillStyle = color;
            ctx.fillRect(x,y,1,1);
        }
    }

    // Stroke in grid lines
    if(grid.length!=64){ // NOT ARC-3
        ctx.beginPath();
        ctx.strokeStyle = '#555555';
        ctx.lineWidth = 1 / scale;
        for(let x=1; x<grid[0].length; x++){
            ctx.moveTo(x, 0);
            ctx.lineTo(x, grid.length);
        }
        for(let y=1; y<grid.length; y++){
            ctx.moveTo(0, y);
            ctx.lineTo(grid[0].length, y);
        }
        ctx.stroke();
    }

    // Gimme
    return canvas;

}



/////////////////////////////////////////////////////////
// Let's begin. Load a specified level. /////////////////
/////////////////////////////////////////////////////////

const SHOW_LEVEL = async (id)=>{
    if(id=="_") return; // "_"? SKIP
    let levelData = await fetchLevelByID(id);
    window.location.hash = window.levelID; // PUT INTO URL
    illustrateLevel(window.levelID); // ILLUSTRATE ME!
}

// LOAD A LEVEL!
{
    // Either the one in the hash, or "jump up"
    let levelID;
    if(window.location.hash){
        levelID = window.location.hash.slice(1); // Remove #
    }else{
        // BY DEFAULT...
        //levelID = '1caeab9d'; // the "SHIFT TO BLUE" level
        //levelID = 'e3721c99'; // the "COLOR MY HOLES" level
        levelID = '0d3d703e'; // recolor
    }

    // Find which dataset contains this level, if any
    let theLevelSet = null;
    for(const [setName, levels] of Object.entries(window.levelSets)){
        if(levels.some(level => level[1]==levelID)){
            theLevelSet = setName;
            break;
        }
    }
    if(theLevelSet){
        // Found it? Set the LevelSet's list
        $('#level_set').value = theLevelSet;
        $('#level_set').onchange();
        // Click the matching item so the index gets tracked too
        const levelSet = window.levelSets[theLevelSet];
        const matchingIndex = levelSet.findIndex(([name, id]) => id === levelID);
        if(matchingIndex >= 0){
            $('#level_select').children[matchingIndex].click();
        }else{
            SHOW_LEVEL(levelID);
        }
    }else{
        // You messed up
        alert(`Whoops! Level ${levelID} not found, please double-check`);
    }
}


/////////////////////////////////////////////////////////
// Click to solve ///////////////////////////////////////
/////////////////////////////////////////////////////////

// Kaggle-style partial credit, like BLAST_EM's score_level():
// each level adds the fraction of its test inputs that pass (attempt_1 OR attempt_2), errors add 0
window.countScore = 0;

const solveOneLevel = async (levelID)=>{

    // "_"? SKIP
    if(levelID=="_") return;

    // Level Data not loaded? WAIT
    if(!window.levelDatas[levelID]){
        await fetchLevelByID(levelID);
    }
    
    // POST levelData to the Flask server
    const response = await fetch('/api/solve?level_id='+encodeURIComponent(levelID),{
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(window.levelDatas[levelID])
    });
    
    // And when get an answer back...
    const levelSolution = await response.json();

    // Saving + checkmarking live in recordLevelSolution() (just below), so the batch solver can reuse it
    recordLevelSolution(levelID, levelSolution);

}

// Save a level's solution, and show ✅/❌ per test in the level select UI
const recordLevelSolution = (levelID, levelSolution)=>{

    // Save it to solutions!
    window.levelSolutions[levelID] = levelSolution;
    
    // HOW MANY TESTS PASS?
    let testsPassed = 0; // For partial credit
    let checkmarks = '';
    let levelData = window.levelDatas[levelID];
    for(let i=0; i<levelData.test.length; i++){
        if(levelSolution.error){
            checkmarks += "❌";
        }else{
            let correctAnswer = levelData.test[i].output.toString(),
                attempt1 = levelSolution.test_solutions[i].attempt_1.toString(),
                attempt2 = levelSolution.test_solutions[i].attempt_2.toString();
                //reasoning1 = levelSolution.test_solutions[i].attempt_1.reasoning,
                //reasoning2 = levelSolution.test_solutions[i].attempt_2.reasoning;
            if( correctAnswer==attempt1 ){
                checkmarks += "✅";
                testsPassed++;
            }else if( correctAnswer==attempt2 ){
                checkmarks += "✅";
                testsPassed++;
            }else{
                // Neither attempt is correct
                checkmarks += "❌";
            }
        }
    }
    // Add this level's fraction of passing tests (once per call, so no double-counting).
    window.countScore += testsPassed / levelData.test.length;
    $(`#passes_${levelID}`).innerText = checkmarks; // show me in the level select UI

}

$('#solve_one').onclick = async ()=>{
    
    // Solve button deactivated...
    $('#solve_one').innerText = '🤔 thinking...';
    $('#solve_one').disabled = true;

    // Try/catch: even if the request blows up (server crashed, non-JSON error page...),
    // the button still gets re-activated instead of being stuck on "thinking..." forever
    try{

        // Wait for Server to solve it...
        await solveOneLevel(window.levelID);

        // Illustrate if worked & there's no error
        if(window.levelSolutions[window.levelID] && !window.levelSolutions[window.levelID].error){
            illustrateLevel(window.levelID);
        }

    }catch(error){
        console.error(error);
    }

    // Solve button re-activated...
    $('#solve_one').disabled = false;
    $('#solve_one').innerText = 'solve this level'

}

const ESTIMATED_SECONDS_PER_LEVEL = 30;
window.batchRunning = false;
window.batchAbortController = null;

$("#solve_all").onclick = async ()=>{

    // Already running? Then this click means STOP (the loop below notices, and wraps up)
    if(window.batchRunning){
        window.batchAbortController.abort();
        return;
    }

    // Every level ID in this list, once each ("_" = skip, and duplicates only get solved once)
    let nameAndIDs = window.levelSets[window.currentLevelSet];
    const levelIDs = [...new Set( nameAndIDs.map(([name,id])=>id).filter(id=>id!="_") )];

    // Button stays clickable (so you can stop it), and shows progress
    $('#solve_all').innerText = '🤔 starting... (click to stop)';
    window.batchRunning = true;
    window.batchAbortController = new AbortController();
    window.countScore = 0; // Partial-credit score, reset per batch
    let levelsDone = 0;
    let doneByTime = '?';

    try{

        // Load all the levels' data, then POST them all to the batch solver in one go
        const levels = {};
        for(const levelID of levelIDs){
            levels[levelID] = await fetchLevelByID(levelID);
        }
        const response = await fetch('/api/solve_batch',{
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(levels),
            signal: window.batchAbortController.signal
        });

        // Read the reply bit by bit: one JSON object per line, as each level finishes
        const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
        let unfinishedText = '';
        while(true){
            const {value, done} = await reader.read();
            if(done) break;
            unfinishedText += value;
            const lines = unfinishedText.split('\n');
            unfinishedText = lines.pop(); // The last bit may be half a line: keep it for next time
            for(const line of lines){
                if(line.trim()=='') continue; // Blank line = the server's heartbeat, ignore

                // First line: how many CPUs the server solves on, for the "done by" estimate
                const message = JSON.parse(line);
                if(message.workers){
                    const estimatedSeconds = Math.ceil(levelIDs.length / message.workers) * ESTIMATED_SECONDS_PER_LEVEL;
                    doneByTime = new Date(Date.now() + estimatedSeconds*1000)
                        .toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', hour12:false });
                }else{
                    // A level's done! Checkmark it, and redraw it if it's the one on screen
                    recordLevelSolution(message.level_id, message);
                    if(message.level_id==window.levelID) illustrateLevel(window.levelID);
                    levelsDone++;
                }
                $('#solve_all').innerText = `🤔 ${levelsDone}/${levelIDs.length}, done by ~${doneByTime} (click to stop)`;
            }
        }

    }catch(error){
        // Clicking stop lands here too (as an AbortError), which is fine
        if(error.name!='AbortError') console.error(error);
    }

    // Solve button back to normal, with the score so far
    window.batchRunning = false;
    const stoppedEarly = levelsDone < levelIDs.length;
    // Partial-credit score, shown with 2 decimals, like "12.50/48 = 26%"
    $('#solve_all').innerHTML = `${stoppedEarly ? 'stopped!' : 'done!'} <b>${window.countScore.toFixed(2)}/${levelsDone} = ${Math.round(100 * (window.countScore/Math.max(levelsDone,1)) )}%</b>`
        + (stoppedEarly ? ` (${levelIDs.length-levelsDone} not solved)` : '');

}

// Leaving (or reloading) the page while a batch runs? Tell the server to stop it.
// sendBeacon still gets sent while the page is closing, unlike a normal fetch.
window.addEventListener('pagehide', ()=>{
    if(window.batchRunning) navigator.sendBeacon('/api/cancel_batch');
});

const _syntaxify = (text)=>{
    // Two spaces for tabs
    text = text.replace(/\t/gi, '    ');
    // All "cN" is highlighted with actual color
    text = text.replace(/"c(\d+)"/gi, `<span class='color-$1'>"c$1"</span>`);
    text = text.replace(/'c(\d+)'/gi, `<span class='color-$1'>'c$1'</span>`);
    // Evolution CoT sections highlighted black/gray
    text = text.replace(/((?<!#)##\s[^\n]*)/gi, `<span class='color-5 cot-header'>$1</span>`);
    text = text.replace(/((?<!#)#\s[^\n]*)/gi, `<span class='color-0 cot-header'>$1</span>`);
    // Gimme
    return text
}