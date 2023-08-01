function convertDecimalToBinary(value, bits = 10) {
    let binaryValues = [];
    let counter = 0;
    while (counter < bits) {
        binaryValues[counter++] = parseInt(value % 2);
        value = parseInt(value / 2);
    }
    return binaryValues;
}

export default {
    template: `
    <div :id="id" style="position:relative"> <!-- :style="jp_props.style" :class="jp_props.classes" -->
    <div class="top-bar">
        <div class="tab">
        </div>
        <a href="#" class="expand-collapse-button" title="separate tracks">
            <i class="material-icons">unfold_more</i>
        </a>
    </div>
        <div class="canvas-wrapper-wrapper">
          <div class="canvas-wrapper">
              <svg width="4096" height="800" class="notestd"/>
              <svg width="4096" height="800" class="notestd"/>
              <svg width="4096" height="800" class="notestd"/>
              <svg width="4200" height="800" class="axes">
                  <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="0" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" />
                    </marker>
                  </defs>
              </svg>
          </div>
          <div class="side-controls">
            <a href="#" class="direction-button" title="direction">
                <i class="material-icons">arrow_forward</i>
            </a>
            <a href="#" class="diffuse-button">
                <i class="material-icons" title="diffuse">brush</i>
            </a>
            <a href="#" class="copy-button" title="copy">
               <i class="material-icons">content_copy</i>
            </a>
          </div>
        </div>
        <div class="pianoroll-controls">
          <a href="#" class="play-button">
            <i class="material-icons" title="play/pause">play_arrow</i>
          </a>
          <a href="#" class="stop-button">
            <i class="material-icons" title="stop">stop</i>
          </a>

          <a href="#" class="undo-button">
            <i class="material-icons" title="undo">undo</i>
          </a>
          <a href="#" class="download-button">
            <i class="material-icons" title="download as MIDI">file_download</i>
          </a>
          <span>Melody Instrument:</span><select class="select select-melody"></select>
          <span>Bass Instrument:</span><select class="select select-bass"></select>
        </div>
        <div class="pianoroll-controls">
          <a href="#" class="mask-selection-button">
            <i class="material-icons" title="mask selection">crop_landscape</i>
          </a>
          <a href="#" class="select-all-button">
            Select
          </a>
          <select class="select-shortcut">
            <option value="all">All</option>
            <option value="melody">Melody</option>
            <option value="bass">Bass</option>
            <option value="drums">Drums</option>
          </select>
          <span style="margin: 20px;">
              Selection Mode
              <select class="select-selection-mode">
                <option value="default">rectangle</option>
                <option value="horizontal">horizontal</option>
              </select>
          </span>
          
          <span style="margin: 20px;">
            Transpose
            <a href="#" class="transpose-up">
                <i class="material-icons" title="transpose up">keyboard_arrow_up</i>
            </a>
            <a href="#" class="transpose-down">
                <i class="material-icons" title="transpose down">keyboard_arrow_down</i>
            </a>
          </span>
          <span style="margin: 20px;">
            Time-Shift
            <a href="#" class="shift-left">
                <i class="material-icons" title="shift left">keyboard_arrow_left</i>
            </a>
            <a href="#" class="shift-right">
                <i class="material-icons" title="shift right">keyboard_arrow_right</i>
            </a>
          </span>
        </div>
    </div>
  `,
    mounted() {
        const MIN_PITCH = 20, MAX_PITCH = 110;
        const CAPACITY = 8;
        const AXES_PAD = 60;
        const TICK_FONT_SIZE = 12;
        const ACTIVE_NOTE_RGB = '0, 255, 0';
        const NOTE_RGB = ['255, 0, 0', '0, 0, 255', '0, 0, 0'];

        if (this.$props.side === 'right') {
            $("#" + this.$props.id + " .side-controls").hide();
        }

        const sleep = ms => new Promise(r => setTimeout(r, ms));

        this.other_pianoroll_idx = 0;
        if( !window.pianorolls) {
            window.pianorolls = [];
            this.other_pianoroll_idx = 1;
        }

        window.pianorolls.push(this);

        this.svgs = $("#" + this.$props.id + " svg.notestd");
        this.axes = $("#" + this.$props.id + " svg.axes")[0];

        const clearVisualizers = () => {
            if (this.visualizers) {
                for (let k in this.visualizers)
                    this.visualizers[k].clearActiveNotes();
            }
        }

        this.player = new mm.SoundFontPlayer('/soundfonts/essentials_sforzando', mm.Player.tone.Master, null, null, {
            run: (note) => {
                this.visualizers[note.program].redraw(note);
            },
            stop: () => {
                $("#" + this.$props.id + " .play-button i").text('play_arrow');
                this.playing = false;
            }
        });
        this.playing = false;
        this.expanded = false;

        $("#" + this.$props.id + " .play-button").click(async () => {
            this.origin = null;
            this.selectedArea = null;
            this.updateNoteSeq();
            if (this.playing) {
                $("#" + this.$props.id + " .play-button i").text('play_arrow');
                this.playerSeconds += this.player.currentPart.context.transport.seconds;
                this.player.stop();
                this.playing = false;
            } else if (this.noteSeq && this.noteSeq.notes.length) {
                this.player.start(this.noteSeq, 120, this.playerSeconds);
                $("#" + this.$props.id + " .play-button i").text('pause');
                this.playing = true;
                while(!this.player.currentPart)
                    await sleep(100);
                this.syncCursor();
            }
            return false;
        });
        this.removeCursor = () => {
            if (this.cursor.active) {
                this.cursor.active = false;
                this.axes.removeChild(this.cursor);
            }
        }
        $("#" + this.$props.id + " .stop-button").click(() => {
            this.origin = null;
            this.selectedArea = null;
            this.player.stop();
            clearVisualizers();
            $("#" + this.$props.id + " .play-button i").text('play_arrow');
            this.playing = false;
            this.removeCursor();
            window.pianorolls[this.other_pianoroll_idx].removeCursor();
            this.playerSeconds = 0;
            this.selectionRect.setAttribute('width', 0);
            this.selectionRect.setAttribute('height', 0);
            this.selectionMode = false;
            drawSelection(false);
            return false;
        });

        $("#" + this.$props.id + " .mask-selection-button").click(() => {
            this.origin = null;
            this.mask();
            return false;
        });
        $("#" + this.$props.id + " .expand-collapse-button").click(() => {
            this.expanded = !this.expanded;

            if(this.expanded) {
                $("#" + this.$props.id + " .expand-collapse-button i").text('unfold_less');
            } else {
                $("#" + this.$props.id + " .expand-collapse-button i").text('unfold_more');
            }
            this.updateNoteSeq();
            this.drawNotes();
            drawSelection(false);
            return false;
        });

        $("#" + this.$props.id + " .undo-button").click(() => {
            this.origin = null;
            this.selectedArea = null;
            TENSOR_STORE.undo(this.active_index);
            this.update();
            return false;
        });

        this.direction = false;
        $("#" + this.$props.id + " .direction-button").click((eventData) => {
            this.direction = !this.direction;
            eventData.currentTarget.children[0].textContent = this.direction ? 'arrow_back' : 'arrow_forward';
        });

        $("#" + this.$props.id + " .copy-button").click(() => {
            let source = this.direction ? window.pianorolls[this.other_pianoroll_idx] : this;
            let dest = this.direction ? this: window.pianorolls[this.other_pianoroll_idx];

            dest.mergeActive(source.getActiveTensor());
            this.update();
        });

        const diffuseButton = $("#" + this.$props.id + " .diffuse-button");
        this.diffuseHandler = () => {
            diffuseButton.addClass('disabled');
            this.can_diffuse = false;
            this.$emit('diffuse', {
                direction: this.direction, tensor: window.pianorolls[this.direction * 1].getActiveTensor(),
                target_slot: window.pianorolls[(!this.direction) * 1].active_index
            });
            return false;
        };
        diffuseButton.click(this.diffuseHandler);

        $("#" + this.$props.id + " .select-all-button").click(() => {
            this.selectionMode = $("#" + this.$props.id + " .select-shortcut").val();
            drawSelection(false);
        });

        const PROGRAMS = [1, 33, 0];
        const DRUM_PITCHES = [36, 38, 42, 46, 45, 48, 50, 49, 51];
        const popSelects = async () => {
            const response = await (await fetch( `/soundfonts/essentials_sforzando/soundfont.json`)).json();
            const instruments = Object.values(response.instruments);
            const selects = $("#" + this.$props.id + " .select");
            for (let i = 0; i < selects.length; i++) {
                let select = selects[i];
                select.innerHTML = instruments.map((e, j) => `<option ${j === PROGRAMS[i] ? 'selected' : ''}>${e}</option>`).join('');
            }
        };

        popSelects();

        const rectHorSelect = $("#" + this.$props.id + " .select-selection-mode");
        this.rectMode = "default";
        rectHorSelect.change(() => this.rectMode = rectHorSelect.val());

        const instChangeHandler = (i, track) => {
            const handler = () => {
                const sel = $("#" + this.$props.id + " .select-" + track + " option:selected");
                PROGRAMS[i] = sel[0].index;
                this.updateNoteSeq();
            };
            return handler;
        };


        $("#" + this.$props.id + " .select-melody").change(instChangeHandler(0, 'melody'));
        $("#" + this.$props.id + " .select-bass").change(instChangeHandler(1, 'bass'));


        this.active_index = 0;
        this.getActiveTensor = () => TENSOR_STORE.getActive(this.active_index);
        this.setActiveTensor = tensor => TENSOR_STORE.setActive(this.active_index, tensor);

        this.mergeActive = tensor => TENSOR_STORE.mergeActive(this.active_index, tensor);

        for (let i = 0; i < CAPACITY; i++) {
            $("#" + this.$props.id + " .tab").append(
                '<button className="tablinks">' + (i + 1) + '</button>');
            $("#" + this.$props.id + " .tab button").last().mousedown((e) => {
                activateTab(i);
            });
        }

        this.updateNoteSeq = () => {
            let result = {
                notes: [],
                quantizationInfo: {stepsPerQuarter: 4},
                tempos: [{time: 0, qpm: 120}],
                totalQuantizedSteps: this.getActiveTensor().length,
                totalTime: 128
            };

            const MAGENTA_PITCH_OFFSET = 19;
            for (let t = 0; t < this.getActiveTensor()[0].length; t++) {
                let pp = 0;
                let pt = 0
                for (let i = 0; i < this.getActiveTensor().length; i++) {
                    let p = this.getActiveTensor()[i][t];
                    if (t === 2) {
                        if (p === this.$props.mask_ids[t]) p = 0;
                        const bits = convertDecimalToBinary(p);
                        for (let b = 0; b < bits.length; b++) {
                            if (bits[b]) {
                                result.notes.push({
                                    pitch: DRUM_PITCHES[b],
                                    quantizedStartStep: i,
                                    quantizedEndStep: i + 1,
                                    program: PROGRAMS[t],
                                    isDrum: true,
                                    velocity: 85
                                });
                            }
                        }
                    } else {
                        if (p === this.$props.mask_ids[t]) p = 1;
                        if (p || i === this.getActiveTensor().length - 1) {
                            if (pp > 1) {
                                result.notes.push({
                                    pitch: pp + MAGENTA_PITCH_OFFSET,
                                    quantizedStartStep: pt,
                                    quantizedEndStep: i,
                                    program: PROGRAMS[t],
                                    velocity: 85
                                });
                            }
                            pp = p;
                            pt = i;
                        }
                    }
                }
            }

            this.noteSeq = result;

            const midi = mm.sequenceProtoToMidi(this.noteSeq);
            const file = new Blob([midi], {type: 'audio/midi'});
            const url = URL.createObjectURL(file);

            let download = $("#" + this.$props.id + " .download-button")[0];
            download.href = url;
            download.download = 'schmubert.mid';


        }

        this.cursor = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        this.cursor.setAttribute('x1', 0);
        this.cursor.setAttribute('y1', 0);
        this.cursor.setAttribute('x2', 0);
        this.cursor.setAttribute('y2', this.svgs[0].height.baseVal.value);
        this.cursor.setAttribute('stroke', 'black');
        this.cursor.active = false;

        this.playerSeconds = 0;
        this.syncCursor = async (pianoroll = null) => {
            clearVisualizers();
            if(!pianoroll) {
              pianoroll = this;
              window.pianorolls[this.other_pianoroll_idx].syncCursor(this);
            }

            this.axes.append(this.cursor);
            this.cursor.active = true;

            do {
                let x = (pianoroll.player.getPlayState() === "started" ? pianoroll.player.currentPart.context.transport.seconds : 0) + pianoroll.playerSeconds;
                x = x * 4096 / 128 + AXES_PAD;
                this.cursor.setAttribute('x1', x);
                this.cursor.setAttribute('x2', x);
                await sleep(10);
            } while (pianoroll.playing && pianoroll.player.currentPart)
        };

        const copyNsWithoutNotes = (ns) => {
            return {
                notes: [], quantizationInfo: ns.quantizationInfo,
                tempos: ns.tempos, totalQuantizedSteps: ns.totalQuantizedSteps,
                totalTime: ns.totalTime
            }
        };

        const getVoiceMinMaxPitch = (ns) => {
            const pitches = ns.notes.map(n => n.pitch);
            return [Math.min(...pitches), Math.max(...pitches)];
        }

        const drawAxes = (minPitch, maxPitch) => {
            let defs = this.axes.children[0];
            $(this.axes).empty();
            this.axes.append(defs);
            if (this.cursor.active)
                this.axes.append(this.cursor);
            this.axes.append(this.selectionRect);

             let xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            xAxis.setAttribute('x1', AXES_PAD);
            xAxis.setAttribute('x2', this.visualizers[0].width + AXES_PAD);
            xAxis.setAttribute('y1', this.visualizers[0].height);
            xAxis.setAttribute('y2', this.visualizers[0].height);
            xAxis.setAttribute('style', 'stroke: black; stroke-width:2');
            xAxis.setAttribute('marker-end', "url(#arrowhead)");
            this.axes.append(xAxis);

            const xTick = (bar) => {
                let x = 16 * bar * this.SCALE;
                let tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tick.setAttribute('x1', x + AXES_PAD);
                tick.setAttribute('x2', x + AXES_PAD);
                tick.setAttribute('y1', this.visualizers[0].height + AXES_PAD / 3);
                tick.setAttribute('y2', this.visualizers[0].height);
                tick.setAttribute('style', 'stroke: black; stroke-width:1');
                this.axes.append(tick);
                let txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                txt.setAttribute('x', x + AXES_PAD - TICK_FONT_SIZE / 3);
                txt.setAttribute('y', this.visualizers[0].height + AXES_PAD * 2 / 3);
                txt.setAttribute('font-size', TICK_FONT_SIZE);
                txt.innerHTML = bar;
                this.axes.append(txt);
            }

            for (let x = 0; x <= 64; x += 1)
                xTick(x);

           let xAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            xAxisLabel.setAttribute('x', 450);
            xAxisLabel.setAttribute('y', this.visualizers[0].height + AXES_PAD / 2);
            xAxisLabel.setAttribute('font-size', TICK_FONT_SIZE * 2);
            xAxisLabel.innerHTML = "Bars";
            this.axes.append(xAxisLabel);


            if (this.expanded)
                return;

            let yAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            yAxisLabel.setAttribute('transform', 'rotate(-90)');
            yAxisLabel.setAttribute('text-anchor', 'middle');
            yAxisLabel.setAttribute('x', -this.visualizers[0].height / 2);
            yAxisLabel.setAttribute('y', 20);
            yAxisLabel.setAttribute('font-size', TICK_FONT_SIZE * 2);
            yAxisLabel.innerHTML = "MIDI Pitch";
            this.axes.append(yAxisLabel);

            this.axes.setAttribute('height', this.visualizers[0].height + AXES_PAD);

            let yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            yAxis.setAttribute('x1', AXES_PAD);
            yAxis.setAttribute('x2', AXES_PAD);
            yAxis.setAttribute('y2', 20);
            yAxis.setAttribute('y1', this.visualizers[0].height);
            yAxis.setAttribute('style', 'stroke: black; stroke-width:2');
            yAxis.setAttribute('marker-end', "url(#arrowhead)");
            this.axes.append(yAxis);


            const yRange = this.visualizers[0].height;
            const pitchRange = maxPitch - minPitch;
            const yTick = (pitch) => {
                let y = (1 - (pitch - minPitch) / pitchRange) * yRange + this.visualizers[0].config.noteHeight / 2;

                let tick = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                tick.setAttribute('x1', AXES_PAD * 4 / 5);
                tick.setAttribute('x2', AXES_PAD);
                tick.setAttribute('y1', y);
                tick.setAttribute('y2', y);
                tick.setAttribute('style', 'stroke: black; stroke-width:1');
                this.axes.append(tick);
                let txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                txt.setAttribute('x', AXES_PAD / 2);
                txt.setAttribute('y', y + TICK_FONT_SIZE / 3);
                txt.setAttribute('font-size', TICK_FONT_SIZE);
                txt.innerHTML = pitch;
                this.axes.append(txt);
            }

            for (let pitch = Math.floor(minPitch / 10) * 10; pitch < maxPitch; pitch += 5)
                yTick(pitch);

        }

        //todo: get y scale: this.visualizers[0].config.minPitch
        this.drawNotes = () => {
            let maxPitch = MAX_PITCH;
            if (this.noteSeq) {

                this.splitUpNotes = Object.fromEntries(PROGRAMS.map(x => [x, copyNsWithoutNotes(this.noteSeq)]));
                this.noteSeq.notes.map(n => this.splitUpNotes[n.program].notes.push(n));

                if (this.expanded) {
                    let [melMin, melMax] = getVoiceMinMaxPitch(this.splitUpNotes[PROGRAMS[0]]);
                    let [bassMin, bassMax] = getVoiceMinMaxPitch(this.splitUpNotes[PROGRAMS[1]]);
                    let [drumMin, drumMax] = getVoiceMinMaxPitch(this.splitUpNotes[PROGRAMS[2]]);

                    let totalRange = melMax + bassMax + drumMax - melMin - bassMin - drumMin;
                    maxPitch = Math.max(maxPitch, totalRange + MIN_PITCH + 2);

                    for(let i = 0; i < this.splitUpNotes[PROGRAMS[0]].notes.length; i++) {
                        this.splitUpNotes[PROGRAMS[0]].notes[i].pitch += maxPitch - melMax;
                    }

                    const bassMed = Math.floor((bassMin + bassMax) / 2);
                    const med = maxPitch - melMax + melMin - (bassMax - bassMin) / 2 - 1;

                    for(let i = 0; i < this.splitUpNotes[PROGRAMS[1]].notes.length; i++) {
                        this.splitUpNotes[PROGRAMS[1]].notes[i].pitch -= bassMed - med;
                    }

                    for(let i = 0; i < this.splitUpNotes[PROGRAMS[2]].notes.length; i++) {
                        this.splitUpNotes[PROGRAMS[2]].notes[i].pitch -= drumMin - MIN_PITCH - 2;
                    }
                }

                this.visualizers = Object.fromEntries(PROGRAMS.map((p, i) =>
                    [p, new mm.PianoRollSVGVisualizer(this.splitUpNotes[p], this.svgs[i],
                        {
                            pixelsPerTimeStep: 32,
                            minPitch: MIN_PITCH,
                            maxPitch: maxPitch,
                            noteRGB: NOTE_RGB[i],
                            activeNoteRGB: ACTIVE_NOTE_RGB
                        })]));

                const addPauseNote = (i, j, start) => {
                    let pause = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                    pause.setAttribute('x', start * this.SCALE);
                    pause.setAttribute('y', 0);
                    pause.setAttribute('width', (j - start) * this.SCALE);
                    pause.setAttribute('height', this.visualizers[0].config.noteHeight);
                    pause.setAttribute('fill', 'transparent');
                    pause.setAttribute('stroke', 'rgb(' + NOTE_RGB[i] + ')');
                    pause.setAttribute('pause', 'true');
                    pause.setAttribute('startStep', start);
                    pause.setAttribute('stopStep', j);
                    pause.setAttribute('class', 'note');
                    this.svgs[i].append(pause);
                }

                const positionPauses = i => {

                    const DEFAULT_PITCHES = [88, 64, 45];

                    //sort so bisect works in selection
                    let children = $(this.svgs[i].children).sort((a, b) => parseInt(a.getAttribute('x')) - parseInt(b.getAttribute('x')));
                    $(this.svgs[i]).append(children);

                    for(let j = 0; j < this.svgs[i].children.length; j++) {
                        if (this.svgs[i].children[j].getAttribute('pause')) {
                            let s = 0, d = 0;
                            if (j > 0) {
                                s += parseInt(this.svgs[i].children[j - 1].getAttribute('y'));
                                d += 1
                            }
                            if (j < this.svgs[i].children.length - 1 && !this.svgs[i].children[j + 1].getAttribute('pause')) {
                                s += parseInt(this.svgs[i].children[j + 1].getAttribute('y'));
                                d += 1;
                            }

                            this.svgs[i].children[j].setAttribute('y', d !== 0 ? Math.round(s / d) :
                                this.visualizers[PROGRAMS[i]].height - this.visualizers[PROGRAMS[i]].config.noteHeight *  (DEFAULT_PITCHES[i] - MIN_PITCH));
                        }
                    }
                };

                this.SCALE = this.svgs[0].width.baseVal.value / this.getActiveTensor().length;
                //draw melody and bass pauses
                for (let i = 0; i < 2; i++){
                    let start = 0;
                    let j = 0;
                    for (; j < this.getActiveTensor().length; j++){
                        if (this.getActiveTensor()[j][i] > 1) {
                            if (start >= 0 && j - start > 0) {
                                addPauseNote(i, j, start);
                            }
                            start = -1;
                        } else if (this.getActiveTensor()[j][i] === 1)
                            start = j;
                    }
                    if (start >= 0 && j - start > 0) {
                        addPauseNote(i, j, start);
                    }
                    positionPauses(i);
                }
                //draw drum pauses
                let start = 0;
                let j = 0
                for (; j < this.getActiveTensor().length; j++) {
                    if (this.getActiveTensor()[j][2] > 0) {
                            if (start >= 0 && j - start > 0) {
                                addPauseNote(2, j, start);
                            }
                            start = -1;
                        } else if (this.getActiveTensor()[j][2] === 0 && start < 0)
                            start = j;
                }
                if (start >= 0 && j - start > 0) {
                    addPauseNote(2, j, start);
                }
                positionPauses(2);
            }

            drawAxes(MIN_PITCH, maxPitch);
        }

        this.update = (resetSelection=true) => {
            if (resetSelection)
                this.selectedArea = null;
            window.pianorolls[this.other_pianoroll_idx].updateNoteSeq();
            window.pianorolls[this.other_pianoroll_idx].drawNotes();
            this.updateNoteSeq();
            this.drawNotes();
        }

        const getSelectionBounds = () => {
            const ans = this.selectionRect.getBBox();
            ans.x -= AXES_PAD;
            return ans;
        }

        this.mask = () => {
            TENSOR_STORE.next(this.active_index);

            doForHorizSelectedNotes((track, svg, note) => {
                let inSelection = !!this.selectionMode || svg.checkIntersection(note, getSelectionBounds());
                if (inSelection) {
                    let index = parseInt(note.getAttribute("data-index"));
                    let start, end;
                    if (note.getAttribute('pause')) {
                        start = parseInt(note.getAttribute('startStep'));
                        end = parseInt(note.getAttribute('stopStep'));
                    } else {
                        start = this.splitUpNotes[PROGRAMS[track]].notes[index].quantizedStartStep;
                        end = this.splitUpNotes[PROGRAMS[track]].notes[index].quantizedEndStep;
                    }
                    for (; start < end; start++)
                        this.getActiveTensor()[start][track] = this.$props.mask_ids[track];
                }
            });

            this.update();
        };

        const transpose = (delta) => {
            TENSOR_STORE.next(this.active_index);
            let moved = false;
            doForHorizSelectedNotes((track, svg, note) => {
                if (track === 2 || note.getAttribute('pause'))
                    return;
                let inSelection = !!this.selectionMode || svg.checkIntersection(note, getSelectionBounds());
                if (inSelection) {
                    let index = parseInt(note.getAttribute("data-index"));
                    let start = this.splitUpNotes[PROGRAMS[track]].notes[index].quantizedStartStep;
                    let end = this.splitUpNotes[PROGRAMS[track]].notes[index].quantizedEndStep;

                    for (; start < end; start++) {
                        if (2 < this.getActiveTensor()[start][track] &&
                            this.getActiveTensor()[start][track] < (this.$props.mask_ids[track] - delta)) {
                            this.getActiveTensor()[start][track] += delta;
                            moved = true;
                        }

                    }
                }
            });

            if (this.rectMode === "default" && moved)
                this.selectionRect.setAttribute("y", parseInt(
                    this.selectionRect.getAttribute("y")) - delta * this.visualizers[0].config.noteHeight);


            this.update(false);
            drawSelection(true);
        }

        $("#" + this.$props.id + " .transpose-up").click(() => transpose(1));
        $("#" + this.$props.id + " .transpose-down").click(() => transpose(-1));

        const timeShift = delta => {
            TENSOR_STORE.next(this.active_index);
            let moved = false;

            let prevTens = TENSOR_STORE.tensors[this.active_index];
            prevTens = prevTens[prevTens.length - 2];

            for (let track = 0; track < 3; track++)
                for (let i = Math.floor(this.selectionBoundHorizontal.x1 / this.SCALE); i < Math.floor(this.selectionBoundHorizontal.x2 / this.SCALE); i++)
                    this.getActiveTensor()[i][track] = this.$props.mask_ids[track];

            doForHorizSelectedNotes((track, svg, note) => {
                let inSelection = !!this.selectionMode || svg.checkIntersection(note, getSelectionBounds());
                if (inSelection) {
                    let index = parseInt(note.getAttribute("data-index"));

                    let start, end;

                    if (note.getAttribute('pause')) {
                        start = parseInt(note.getAttribute('startStep'));
                        end = parseInt(note.getAttribute('stopStep'));
                    } else {
                        start = this.splitUpNotes[PROGRAMS[track]].notes[index].quantizedStartStep;
                        end = this.splitUpNotes[PROGRAMS[track]].notes[index].quantizedEndStep;
                    }

                    for (; start < end; start++) {
                        if (start + delta >= 0 && start + delta <= 1024) {
                            this.getActiveTensor()[start + delta][track] = prevTens[start][track];
                            moved = true;
                        }

                    }
                }
            });

            if (this.rectMode === "default" && moved) {
                this.selectionRect.setAttribute("x", parseInt(
                    this.selectionRect.getAttribute("x")) + delta * this.SCALE);
                this.selectionBoundHorizontal.x1 += delta * this.SCALE;
                this.selectionBoundHorizontal.x2 += delta * this.SCALE;
            }


            this.update(false);
            drawSelection(true);
        };

        $("#" + this.$props.id + " .shift-left").click(() => timeShift(-16));
        $("#" + this.$props.id + " .shift-right").click(() => timeShift(16));

        const activateTab = i => {
            this.active_index = i;
            this.updateNoteSeq();
            this.drawNotes();
            $("#" + this.$props.id + " button").removeClass("active");
            $($("#" + this.$props.id + " button")[i]).addClass("active");
        };
        activateTab(this.$props.side === 'left' ? 0 : 1);


        //executes callback for notes in horizonal selection
        const doForHorizSelectedNotes = callback => {
            this.svgs.each((i, svg) => {

                if (this.selectionMode === 'melody' && i !== 0 || this.selectionMode === 'bass' && i !== 1 || this.selectionMode === 'drums' && i !== 2)
                    return;

                let children = svg.children;

                let l = 0, h = children.length;
                let upperX = Infinity;
                if (!this.selectionMode && this.selectionBoundHorizontal) {
                    upperX = this.selectionBoundHorizontal.x2;
                    //search horiz. start of selection binary
                    while (h - l > 1) {
                        let m = Math.floor((h + l) / 2);
                        if (parseInt(children[m].getAttribute('x')) > this.selectionBoundHorizontal.x1)
                            h = m - 1;
                        else
                            l = m;
                    }
                }
                for (; l < children.length && parseInt(children[l].getAttribute('x')) < upperX; l++)
                    if (children[l].getAttribute('class') === 'note')
                        callback(i, svg, children[l]);
            });
        }
        const drawSelection = e => {
            if (e) {
                if (e.isTrusted) {
                    let x = this.origin.x;
                    let y = this.origin.y;
                    let width = e.offsetX - this.origin.x;
                    let height = e.offsetY - this.origin.y;

                    if (width < 0) {
                        width *= -1;
                        x -= width;
                    }
                    if (height < 0) {
                        height *= -1;
                        y -= height;
                    }
                    this.selectionRect.setAttribute('x', x);
                    this.selectionRect.setAttribute('width', width);
                    if (this.rectMode === "default") {
                        this.selectionRect.setAttribute('y', y);
                        this.selectionRect.setAttribute('height', height);
                    } else {
                        this.selectionRect.setAttribute('y', 0);
                        this.selectionRect.setAttribute('height', this.visualizers[0].height);
                    }
                }
            } else {
                this.selectionRect.setAttribute('width', 0);
                this.selectionRect.setAttribute('height', 0);
            }

            doForHorizSelectedNotes((track, svg, note) => {
                let inSelection = !!this.selectionMode || svg.checkIntersection(note, getSelectionBounds());
                let fill = note.getAttribute('pause') ? 'transparent' : 'rgb(' + NOTE_RGB[track] + ')';
                note.setAttribute('fill', (inSelection ? 'rgb(' + ACTIVE_NOTE_RGB + ')' : fill));
            });
        };


        this.drawNotes();

        this.axes.addEventListener('click', async e => {
            if (this.selectedArea && (this.selectedArea.w ** 2 > 9 && this.selectedArea.h ** 2 > 9)) {
                return;
            }
            this.playerSeconds = (e.offsetX - AXES_PAD) * 128 / 4096;
            if (this.player.getPlayState() === "started") {
                this.player.stop();
                this.player.start(this.noteSeq, 120, this.playerSeconds);
            }

            this.syncCursor();
        });

        this.origin = null;
        this.selectedArea = null;

        this.selectionRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        this.selectionRect.setAttribute('x', 0);
        this.selectionRect.setAttribute('y', 0);
        this.selectionRect.setAttribute('width', 0);
        this.selectionRect.setAttribute('height', 0);
        this.selectionRect.setAttribute('fill', 'transparent');
        this.selectionRect.setAttribute('stroke', 'black');

        this.axes.addEventListener('mousedown', async e => {
            this.origin = {x: e.offsetX, y: e.offsetY};
            if (this.selectionBoundHorizontal)
                drawSelection(e);
            this.selectionBoundHorizontal = {x1: e.offsetX, x2: e.offsetX};
        });
        this.axes.addEventListener('mouseup', () => {
            this.origin = null;
        });
        this.axes.addEventListener('mousemove', e => {
            if (this.origin) {
                if (this.selectionMode) {
                    this.selectionMode = false;
                    this.drawNotes();
                }

                this.axes.append(this.selectionRect);
                this.selectedArea = {
                    x: Math.min(this.origin.x, e.offsetX) - AXES_PAD, y: Math.min(this.origin.y, e.offsetY),
                    w: Math.abs(e.offsetX - this.origin.x) + AXES_PAD, h: Math.abs(e.offsetY - this.origin.y)
                };
                this.selectionBoundHorizontal.x1 = Math.min(this.selectionBoundHorizontal.x1, this.selectedArea.x);
                this.selectionBoundHorizontal.x2 = Math.max(this.selectionBoundHorizontal.x2, this.selectedArea.x + this.selectedArea.w);
                drawSelection(e);
            }
        });
        this.last_updated = 0;
        this.checkForTensorUpdate = () => {
            if (this.$props.t > this.last_updated) {
                this.last_updated = this.$props.t;
                if (this.$props.target_slot > -1) {
                    TENSOR_STORE.setActive(this.$props.target_slot, this.$props.tensor);
                     $(".diffuse-button").removeClass('disabled');
                     $("#" + window.pianorolls[0].id + ".diffuse-button").click(window.pianorolls[0].diffuseHandler);
                }
                 else
                    this.setActiveTensor(this.$props.tensor);
                this.update();
            }
        }
        this.checkForTensorUpdate();
    },
    updated() {
        this.checkForTensorUpdate();
    },
    methods: {},
    props: {
        mask_ids: Array,
        side: String,
        tensor: Array,
        id: Number,
        t: Number,
        target_slot: Number
    },
};