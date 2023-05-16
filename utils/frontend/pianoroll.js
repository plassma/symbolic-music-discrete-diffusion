function convertDecimalToBinary(value, bits=10) {
   let binaryValues = [];
   let counter = 0;
   while (counter < bits) {
      binaryValues[counter++] = parseInt(value % 2);
      value = parseInt(value / 2);
   }
   return binaryValues;
}

Vue.component("pianoroll", {
  template: `
    <div :id="jp_props.id" style="position:relative" :style="jp_props.style" :class="jp_props.classes">
    <div class="tab">
        
    </div>
        <div class="canvas-wrapper-wrapper">
          <div class="canvas-wrapper">
              <canvas width="4096" height="800"/>
          </div>
          <div class="side-controls">
            <a href="#" class="diffuse-button">
                <i class="material-icons">brush</i>
            </a>
            <a href="#" class="copy-button">
               <i class="material-icons">content_copy</i>
            </a>
          </div>
        </div>
        <div class="pianoroll-controls">
          <a href="#" class="play-button">
            <i class="material-icons">play_arrow</i>
          </a>
          <a href="#" class="stop-button">
            <i class="material-icons">stop</i>
          </a>
          <a href="#" class="mask-selection-button">
            <i class="material-icons">crop_landscape</i>
          </a>
          <a href="#" class="mask-melody-button">
            <i class="material-icons">music_note</i>
          </a>
          <a href="#" class="mask-bass-button">
            <i class="material-icons">music_note</i>
          </a>
          <a href="#" class="mask-drums-button">
            <i class="material-icons">music_note</i>
          </a>
          <a href="#" class="undo-button">
            <i class="material-icons">undo</i>
          </a>
          <a href="#" class="download-button">
            <i class="material-icons">file_download</i>
          </a>
          <select class="select select-melody"></select>
          <select class="select select-bass"></select>
          <select class="select select-drums"></select>
        </div>
    </div>
  `,
  mounted() {
    if(this.$props.jp_props.id === 16) {
      $("#" + this.$props.jp_props.id + " .side-controls").prependTo($("#" + this.$props.jp_props.id + " .canvas-wrapper-wrapper"));
    }

    const sleep = ms => new Promise(r => setTimeout(r, ms));
    comp_dict[this.$props.jp_props.id] = this;
    this.other_pianoroll_id = this.$props.jp_props.id === 14 ? 16:14;
    this.canvas = $("#"+this.$props.jp_props.id + " canvas")[0];
    const ctx = this.canvas.getContext('2d');
    this.player = new mm.SoundFontPlayer('http://127.0.0.1:8097/sgm_plus');//new mm.SoundFontPlayer('https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus');
    this.playing = false;

    $("#"+this.$props.jp_props.id + " .play-button").click(() => {
      this.origin = null;
      this.selectedArea = null;
      this.updateNoteSeq();
      if(this.playing) {
        $("#"+this.$props.jp_props.id + " .play-button i").text('play_arrow');
        this.player.pause();
        this.playing = false;
      } else if (this.noteSeq && this.noteSeq.notes.length) {
        if (this.player.isPlaying()) {
          let t = this.player.currentPart.context.transport.seconds;
          this.player.stop();
          this.player.start(this.noteSeq);
          setTimeout(() => this.player.seekTo(t), 100);
        }
        else {
          this.player.start(this.noteSeq);
        }
         $("#"+this.$props.jp_props.id + " .play-button i").text('pause');
         this.playing = true;
         this.syncCursor();
      }
      return false;});
    $("#"+this.$props.jp_props.id + " .stop-button").click(() => {
      this.origin = null;
      this.selectedArea = null;
      this.player.stop();
      $("#"+this.$props.jp_props.id + " .play-button i").text('play_arrow');
      this.playing = false;
      this.drawNotes();
      comp_dict[this.other_pianoroll_id].drawNotes();
      return false;});

    $("#"+this.$props.jp_props.id + " .mask-selection-button").click(() => {
      this.origin = null;
      this.mask();
      return false;
    });
    $("#"+this.$props.jp_props.id + " .mask-melody-button").click(() => {
      this.origin = null;
      this.selectedArea = {x: 0, y: 0, w: 4096, h: (this.$props.jp_props.options.mask_ids[0] + 1)* SCALE};
      this.mask();
      return false;
    });
    $("#"+this.$props.jp_props.id + " .mask-bass-button").click(() => {
      this.origin = null;
      this.selectedArea = {x: 0, y: (this.$props.jp_props.options.mask_ids[0] + 2) * SCALE, w: 4096, h: (this.$props.jp_props.options.mask_ids[0] + 1) * SCALE};
      this.mask();
      return false;
    });
    $("#"+this.$props.jp_props.id + " .mask-drums-button").click(() => {
      this.origin = null;
      this.selectedArea = {x: 0, y: 2 * TRACK_OFFSET + 3*SCALE, w: 4096, h: 80};
      this.mask();
      return false;
    });
    $("#"+this.$props.jp_props.id + " .undo-button").click(() => {
      this.origin = null;
      this.selectedArea = null;
      this.notes.undo();
      this.drawNotes();
      return false;
    });
    $("#"+this.$props.jp_props.id + " .copy-button").click(() => {
       comp_dict[this.other_pianoroll_id].notes.active = this.notes.active;
       comp_dict[this.other_pianoroll_id].drawNotes();
     });

    $("#"+this.$props.jp_props.id + " .diffuse-button").click(() => {
      const event = {
        event_type: "onDiffuse",
        vue_type: this.$props.jp_props.vue_type,
        id: this.$props.jp_props.id,
        page_id: page_id,
        websocket_id: websocket_id,
        notes: this.notes.diffActiveTensors
      };
      send_to_server(event, "event");
     });

    let PROGRAMS = [1, 33, 0];
    let DRUM_PITCHES = [36, 38, 42, 46, 45, 48, 50, 49, 51];
    const popSelects = async () => {
      const response = await (await fetch(`http://127.0.0.1:8097/sgm_plus/soundfont.json`)).json();
      const instruments = Object.values(response.instruments);
      const selects = $("#" + this.$props.jp_props.id + " .select");
      for(let i = 0; i < selects.length; i++) {
        let select = selects[i];
        select.innerHTML = instruments.map((e, j) => `<option ${j === PROGRAMS[i] ? 'selected' : ''}>${e}</option>`).join('');
      }
    };

    popSelects();

    const instChangeHandler = (i, track) => {
      const handler = () => {
        const sel = $("#" + this.$props.jp_props.id + " .select-" + track + " option:selected");
        PROGRAMS[i] = sel[0].index;
        this.updateNoteSeq();
      };
      return handler;
    };



    $("#" + this.$props.jp_props.id + " .select-melody").change(instChangeHandler(0, 'melody'));
    $("#" + this.$props.jp_props.id + " .select-bass").change(instChangeHandler(1, 'bass'));
    $("#" + this.$props.jp_props.id + " .select-drums").change(instChangeHandler(2, 'drums'));

    const SCALE = 4;
    const TRACK_OFFSET = SCALE * this.$props.jp_props.options.mask_ids[0];
    const DRUM_OFFSET = 5 * SCALE;
    const CAPACITY = 8;
    this.notes = new TensorStore(this.$props.jp_props.options.mask_ids, CAPACITY);

    const activateTab = (i) => {
      $("#" + this.$props.jp_props.id + " button").removeClass("active");
      $($("#" + this.$props.jp_props.id + " button")[i]).addClass("active");
    };

    this.markDiffActive = (f = true) => {
      if(f && comp_dict[this.other_pianoroll_id]) comp_dict[this.other_pianoroll_id].markDiffActive(false);
      $("#" + this.$props.jp_props.id + " button").removeClass("d-active");
      for(let i = 0; i < TensorStore.diffActive.length; i++) {
        if(TensorStore.diffActive[i])
          $($("#" + this.$props.jp_props.id + " button")[i]).addClass("d-active")
      }
    };

    this.markDiffActive();

    for(let i = 0; i < CAPACITY; i++) {
      $("#"+this.$props.jp_props.id + " .tab").append(
          '<button className="tablinks">' + (i + 1) + '</button>');
      $("#"+this.$props.jp_props.id + " .tab button").last().mousedown((e) => {
        if(e.which === 1) {

          if(this.notes.autoDiffActivation) {
            TensorStore.diffActive[this.notes.i] = false;
            TensorStore.diffActive[i] = true;
          }

          this.notes.i = i;
          activateTab(i);
          this.drawNotes();
        } else {
          TensorStore.diffActive[i] = !TensorStore.diffActive[i];
        }
        this.markDiffActive();
      });
    }

    activateTab(0);


    const getCoords = (i, t) => {
      let p = this.notes.active[i][t];
      let y_off = (t + 1) * TRACK_OFFSET + SCALE * t * 2;
      return [(i * SCALE), (y_off - p * SCALE)]
    }

    this.updateNoteSeq = () => {
      let result = {
        notes: [],
          quantizationInfo: {stepsPerQuarter: 4},
          tempos: [{time: 0, qpm: 120}],
          totalQuantizedSteps: this.notes.active.length,
          totalTime:128
          };
      const MAGENTA_PITCH_OFFSET = 19;
      for(let t = 0; t < this.notes.active[0].length; t++) {
        let pp = 0;
        let pt = 0
        for (let i = 0; i < this.notes.active.length; i++) {
          let p = this.notes.active[i][t];
          if (p === this.$props.jp_props.options.mask_ids[t]) p = 0;
          if(t === 2) {
            const bits = convertDecimalToBinary(p);
            for(let b = 0; b < bits.length; b++) {
              if(bits[b]){
                result.notes.push({pitch: DRUM_PITCHES[b], quantizedStartStep: i, quantizedEndStep: i + 1, program: PROGRAMS[t], isDrum: true, velocity: 63});
              }
            }
          } else {
            if (p) {
              if (pp > 1) {
                let v = t > 0 ? 80:63;
                result.notes.push({pitch: pp + MAGENTA_PITCH_OFFSET, quantizedStartStep: pt, quantizedEndStep: i, program: PROGRAMS[t], velocity: v});
              }
              pp = p;
              pt = i;
            }
          }
        }
      }
      this.noteSeq = result;

      const midi = mm.sequenceProtoToMidi(this.noteSeq);
      const file = new Blob([midi], { type: 'audio/midi' });
      const url = URL.createObjectURL(file);

      let download = $("#"+this.$props.jp_props.id + " .download-button")[0];
      download.href = url;
      download.download = 'schmubert.mid';


    }

    this.inSelectedArea = (i, t) => {
      if(! this.selectedArea) return false;
      let x_contained = this.selectedArea.x <= i * SCALE && this.selectedArea.x + this.selectedArea.w >= (i + 1) * SCALE;
      if (!x_contained) return false;
      if(t === 2) return x_contained && this.selectedArea.y + this.selectedArea.h >= t * TRACK_OFFSET + DRUM_OFFSET;
      let [_, y] = getCoords(i, t);
      return this.selectedArea.y <= y && this.selectedArea.y + this.selectedArea.h >= y + SCALE;
    };

    this.drawNotes = () => {
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      let fillStyle;
      for(let t = 0; t < this.notes.active[0].length; t++) {
        switch (t) {
              case 0:
                fillStyle = ["red", "green"];
                break;
              case 1:
                fillStyle = ["blue", "orange"];
                break;
              default:
                fillStyle = ["black", "yellow"];
        }
        if(t < 2) {
          for (let i = 0; i < this.notes.active.length; i++) {
            let [x, y] = getCoords(i, t);
            ctx.fillStyle = fillStyle[this.inSelectedArea(i, t) * 1];
            ctx.beginPath();
            ctx.rect(x, y, SCALE, SCALE);
            ctx.fill();
          }
          } else {
            for(let i = 0; i < this.notes.active.length; i++) {
              const bits = convertDecimalToBinary(this.notes.active[i][t]);
              for(let b = 0; b < bits.length; b++) {
                if(bits[b]) {
                  let y = t * TRACK_OFFSET + DRUM_OFFSET + (b) * SCALE;
                  ctx.fillStyle = fillStyle[this.inSelectedArea(i, t) * 1];
                  ctx.beginPath();
                  ctx.rect(i * SCALE, y, SCALE, SCALE);
                  ctx.fill();
                }
              }
            }
          }
      }
    };

    this.mask = () => {
      this.notes.next();
      for(let t = 0; t < this.notes.active[0].length; t++) {
          for (let i = 0; i < this.notes.active.length; i++) {
          if (this.inSelectedArea(i, t)) {
              this.notes.active[i][t] = this.$props.jp_props.options.mask_ids[t];
          }
        }
      }
      this.selectedArea = null;
      this.drawNotes();
    };

    const drawLine = (x) => {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, 800);
        ctx.stroke();
    };

    const drawSelection = (e) => {
            ctx.strokeStyle = "#000";
            ctx.beginPath();
            ctx.rect(this.origin.x, this.origin.y, e.offsetX - this.origin.x, e.offsetY - this.origin.y);
            ctx.stroke();
    };


    this.syncCursor = async (pianoroll = null) => {

      if(!pianoroll) {
        pianoroll = this;
        comp_dict[this.other_pianoroll_id].syncCursor(this);
      }

      while (!pianoroll.player.currentPart) {
        await sleep(100);
      }
      let x;

      do {
        x = pianoroll.player.currentPart.context.transport.seconds * 4096 / 128;
        this.drawNotes();
        drawLine(x);
        await sleep();
      } while (pianoroll.playing)

      this.drawNotes();
      drawLine(x);
    };

    this.drawNotes();
    this.canvas.ondragstart = () => false;
    this.canvas.addEventListener('click', async e => {
      if (this.selectedArea && (this.selectedArea.w ** 2 > 9 && this.selectedArea.h ** 2 > 9)) {
        return;
      }
      if (!this.player.isPlaying()) {
        if(this.noteSeq) {
          this.player.start(this.noteSeq);
          setTimeout(() => {
            this.player.pause();
            this.player.seekTo(e.offsetX * 128 / 4096);
            this.syncCursor();
            }, 100);
          return;
        }
      }
      setTimeout(() => {
        this.player.seekTo(e.offsetX * 128 / 4096);
        this.syncCursor();
      }, 100);

    });

    this.origin = null;
    this.selectedArea = null;
    this.canvas.addEventListener('mousedown', async e => {
      this.origin = {x: e.offsetX, y: e.offsetY};
    });
    this.canvas.addEventListener('mouseup', () => {
      this.origin = null;
    })
    this.canvas.addEventListener('mousemove', e => {
      if(this.origin) {
        this.selectedArea = {x: Math.min(this.origin.x, e.offsetX), y: Math.min(this.origin.y, e.offsetY),
        w: Math.abs(e.offsetX - this.origin.x), h: Math.abs(e.offsetY - this.origin.y)};
        this.drawNotes();
        drawSelection(e);
      }
    });

    const sendConnectEvent = () => {
      if (websocket_id === "") return;
      const event = {
        event_type: "onConnect",
        vue_type: this.$props.jp_props.vue_type,
        id: this.$props.jp_props.id,
        page_id: page_id,
        websocket_id: websocket_id,
      };
      send_to_server(event, "event");
      clearInterval(connectInterval);
    };
    const connectInterval = setInterval(sendConnectEvent, 100);

    this.t = 0;
    if (this.$props.jp_props.options.notes.t > this.t) {
      this.notes.diffActiveTensors = this.$props.jp_props.options.notes.tensor;
      this.t = this.$props.jp_props.options.notes.t;
    }

  },
 updated() {

    if (this.$props.jp_props.options.notes.t > this.t) {
      this.notes.diffActiveTensors = this.$props.jp_props.options.notes.tensor;
      this.t = this.$props.jp_props.options.notes.t;
    }

    this.selectedArea = null;
    this.drawNotes();
    this.updateNoteSeq();

  },
  methods: {

  },
  props: {
    jp_props: Object,
  },
});
