import Foundation

/// Explicit editorial sections, retaining step IDs and the shared block/reading contract.
@MainActor enum FeatureLessonReview {
    static func refine(_ content:LessonContent)->LessonContent {
        var result=content
        switch content.entryID {
        case "prepare-media":
            replace("try.4", content:&result, blocks:[
                .init(id:"try.4.body",kind:.canvasAction,title:"Match the model input",text:"Before adding a model, select each Prepare media node and compare its captured dtype, sample shape and axis order. In Train’s Data mapping choose Prepared tensor and that exact column. A required mask is a separate named input."),
                .init(id:"try.4.observation",kind:.observation,title:"Carry the recipe forward",text:"Saving a model preserves its expected tensor shape. Reuse the same preparation for new files; the package does not become an automatic raw-file reader."),
                .init(id:"try.4.detail",kind:.detail,title:"Other audio paths",text:"For speech-to-text, use Transcribe a private recording and its connected Load model processor. For spectral analysis, Prepare media offers Magnitude, Power and Log-mel. FFT size controls the number of samples examined together; frame hop controls the gap between those examinations. The tensor uses Channels × frames × frequency. This practice records no microphone input and downloads no model.")
            ])
        case "document-classification":
            replace("try.4", content:&result, blocks:[
                .init(id:"try.4.body",kind:.canvasAction,title:"Save and reuse the fitted pipeline",text:"1. Select [Save classifier and text preparation](kwip-node://control/save), keep Portable package and choose a new destination. Choose Save model.\n2. On [Reload classifier package](kwip-node://control/load), choose Prebuilt / document model package and select that folder.\n3. Select [Classify unseen documents](kwip-node://control/predict), then Run Flow.\n4. Inspect [Inspect document predictions](kwip-node://data/predictions). Save/reopen the project and predict again."),
                .init(id:"try.4.observation",kind:.observation,title:"Check the new population",text:"The supplied [Three unseen documents](kwip-node://data/unseen) have no answer labels. Prediction reuses the fitted vocabulary, IDF, class order and coefficients; it must not train again or require the original training folder. Raw training documents are not bundled."),
                .init(id:"try.4.detail",kind:.detail,title:"Evaluate and deliver later",text:"Evaluate labeled documents measures a separate labeled population. Its predictions use the standard Output table; Metrics is separate. Review thresholds flag low model probabilities without changing the predicted class. Connect Table Output for an explicit CSV, database or API write. Viewing History never repeats that write.")
            ])
        case "self-play":
            replace("configure", content:&result, blocks:[
                .init(id:"configure.body",kind:.explanation,title:"One learning iteration",text:"An iteration is one repeat of the learning loop. First, the current fixed model searches possible moves and plays games. Completed games supply examples: which actions search preferred and how the game ended. Those examples enter replay, the bounded collection used for later learning updates."),
                .init(id:"configure.canvasAction",kind:.canvasAction,title:"Inspect the actual loop",text:"Open [Learn Self-play](kwip-node://control/train) → Configure Self-play. Review Self-play, Search, Learning and Replay & checkpoints. Validate bindings checks the rules/model without learning. Apply saves settings."),
                .init(id:"configure.observation",kind:.observation,title:"Distinguish games from updates",text:"Games collect examples; updates change learned numbers. A truncated or failed game is recorded, but supplies no invented draw target. Each committed boundary preserves the learning state needed for continuation.")
            ])
            replace("compare", content:&result, blocks:[
                .init(id:"compare.body",kind:.canvasAction,title:"Compare frozen opponents",text:"In Configure Self-play → Evaluation, select Initial model or Random legal actions first. Compare both player seats and complete seat pairs in [Inspect evaluation](kwip-node://data/evaluation). Keep the rules, openings and search allowance fixed when comparing runs. Count failed/truncated attempts separately."),
                .init(id:"compare.explanation",kind:.explanation,title:"What the two modes measure",text:"Policy-only chooses from the model’s scores directly. Search-assisted looks ahead through legal moves before choosing. Root noise is random variation added during collection to explore alternatives; evaluation disables it. A lower loss or wins against random do not establish strong play."),
                .init(id:"compare.detail",kind:.detail,title:"Stronger comparisons and compute",text:"Uniform-prior search explores with equal starting action preferences; a retained checkpoint supplies an earlier frozen model. Held-out openings are legal move sequences excluded from learning replay. In Compute, CPU still runs rules/search decisions. Apple GPU may perform batched model evaluation and learning after actual graph/optimizer validation. GPU recovery is numerically qualified, not bit-identical CPU continuation. Automatic keeps CPU unless the complete workload has a verified speed advantage.")
            ])
        case "mixture-experts":
            replace("try.0", content:&result, blocks:[
                .init(id:"try.0.body",kind:.canvasAction,title:"Trace an expert on this canvas",text:"Expand [Feature mixture](kwip-node://model/mixture). Find Expert B and follow its Linear → GELU → Linear chain. Select a layer to use its normal Configuration controls. Each expert accepts two measurements and returns two scores. Keep wires inside that expert’s chain; Undo reverses a canvas edit."),
                .init(id:"try.0.observation",kind:.observation,title:"Read the boundary connections",text:"Dashed rose links carry the routed input to each chain and combine its returned scores. Ordinary layer wires stay teal. The container selects and combines experts; its boundary is not another hidden canvas.")
            ])
            replace("try.1", content:&result, blocks:[
                .init(id:"try.1.body",kind:.canvasAction,title:"Inspect which experts receive a row",text:"Open [Feature mixture](kwip-node://model/mixture) → Configure mixture → Routing. Keep two experts per unit and normalized selected weights. Router layers shows the input, gate and expert-score boundary; select a layer to inspect its connections and fields."),
                .init(id:"try.1.explanation",kind:.explanation,title:"Understand the router",text:"The router scores each expert for one observation. Top-k chooses the k highest scores: top-2 chooses two experts here. Normalized weights combine their outputs into one result. Temperature changes how sharply those scores favor one expert."),
                .init(id:"try.1.detail",kind:.detail,title:"Top-1 and topology",text:"With normalized top-1, the selected expert receives weight one, so the task loss alone cannot teach the gate which expert to choose. The explicit balancing objective can still affect it. Advanced router JSON is optional for topology changes; it is not needed for this practice.")
            ])
            replace("try.3", content:&result, blocks:[
                .init(id:"try.3.body",kind:.canvasAction,title:"Compare the measured objectives",text:"Select [Fit and evaluate classifier](kwip-node://control/train). In Metrics choose MoE objective and compare Task loss, Balance penalty, Router z penalty and their sum at completed reporting intervals. Best Validation still selects using the task metric."),
                .init(id:"try.3.explanation",kind:.explanation,title:"Separate goals",text:"Task loss measures prediction error. Balance penalty discourages concentrating all traffic on a few experts. Router z penalty restrains the scale of router scores. Their weighted sum trains the model; a smaller sum alone does not prove better validation predictions."),
                .init(id:"try.3.question",kind:.question,title:"Check your interpretation",text:"If expert traffic becomes more even while Validation gets worse, has the task improved? Compare a dense model with the same held-out observations before concluding that routing helps.")
            ])
        case "fine-tuning":
            replace("try.1", content:&result, blocks:[
                .init(id:"try.1.body",kind:.canvasAction,title:"Use prompt/response only with matching data",text:"The supplied CSV contains only text. For your own table with two String columns, choose Prompt + response in [Adapt the loaded model](kwip-node://control/train) → Configure fine-tuning → Data & loss. Set both column names and an explicit separator. Choose Response tokens only when the prompt supplies context and the answer supplies the learning target."),
                .init(id:"try.1.observation",kind:.observation,title:"Inspect the training positions",text:"Validate & inspect shows a bounded first token window and checks a real forward/backward pass without updating weights. Prompt and padding positions are excluded from response-only loss. Overlong responses are refused rather than silently removed."),
                .init(id:"try.1.callout",kind:.callout,title:"Inspection can reveal text",text:"This explicit inspection may display training text. Its window is a preview, not a sample that chooses which rows train.")
            ])
            replace("try.5", content:&result, blocks:[
                .init(id:"try.5.body",kind:.canvasAction,title:"Choose retained recovery boundaries",text:"In [Adapt the loaded model](kwip-node://control/train) → Resume & retention, choose the number and byte allowance of completed boundaries to keep. Zero preserves legacy terminal-only behavior. A positive count retains completed epochs atomically."),
                .init(id:"try.5.observation",kind:.observation,title:"Pause is different from Stop",text:"Pause after completed epoch finishes that epoch and supplies selected inference weights. Stop now does not complete the run. Only already committed retained epochs can recover."),
                .init(id:"try.5.detail",kind:.detail,title:"Exact continuation requirements",text:"Exact CPU resume requires matching base, tokenizer, data, settings, runtime and original schedule horizon. Best Validation weights may come from an earlier epoch than the latest optimizer state. MPS supports weights reuse with fresh optimizer memory, not exact resume.")
            ])
            replace("try.6", content:&result, blocks:[
                .init(id:"try.6.body",kind:.canvasAction,title:"Choose what Save model should retain",text:"Open [Save the updated model](kwip-node://control/save-model) → Configure save. Choose the purpose you need and a new unused folder. Standalone inference saves merged LoRA weights; Adapter + base identity needs that exact base for reuse; CPU Training resume package retains the latest optimizer, schedule and random state."),
                .init(id:"try.6.observation",kind:.observation,title:"Keep the matching inputs",text:"Raw documents are not included. To continue a CPU resume package, import it from Train, connect the matching base and retain the original Training/Validation data. Continuing is not the same as starting again from selected inference weights."),
                .init(id:"try.6.detail",kind:.detail,title:"Code boundary",text:"Training Code can replay the captured task using explicit external model, data, resource and optional resume-package bindings. Whole-flow Generate/Save/Evaluate replay is refused rather than silently omitted.")
            ])
        default:break
        }
        if let reflection=reflections[content.entryID], let index=result.steps.firstIndex(where:{$0.id=="inspect"}) {
            let boundary=result.steps[index].blocks.first?.text ?? ""
            result.steps[index].title="Explain the evidence"
            result.steps[index].blocks=[.init(id:"inspect.body",kind:.question,title:reflection.0,text:reflection.1),.init(id:"inspect.scope",kind:.detail,title:"Scope and retention",text:boundary)]
        }
        return result
    }
    private static func replace(_ id:String,content:inout LessonContent,blocks:[LessonContentBlock]) {
        guard let index=content.steps.firstIndex(where:{$0.id==id}) else{return}
        content.steps[index].blocks=blocks
    }
    private static let reflections:[String:(String,String)] = [
        "prepare-media":("Compare source and prepared values","Why does the last 256-sample audio window contain 232 real samples and 24 padding values? Use its valid mask to distinguish them. Changing display scale must not change the captured model input."),
        "document-classification":("Explain the held-out result","What was learned from Training, and what stayed fixed for the three unseen documents? Inspect a predicted label alongside its source text. A good score on this synthetic fixture does not establish quality on your own documents."),
        "extract-text":("Check recognized text against pixels","Compare INVOICE 42 with its captured image and source-region boxes. How do you distinguish no text found from a failed read? A high recognition score is not proof that a field was interpreted correctly."),
        "game-rules":("Explain a test result","For Take 3 from 5, predict remaining and to_move, then compare the actual test and trace. An illegal move must leave the previous state intact. Passing your authored tests is evidence about those cases, not proof that every possible game is correct."),
        "game-episodes":("Explain what changed","Did the model learn during this run, or only choose moves from fixed scores? Compare terminal outcomes with moveLimit rows. A limit is not a draw."),
        "transcribe-audio":("Review the recording","Listen to a captured segment and compare its text, names and time interval. Does the model omit words or invent speech during silence? Recorded text and timestamps need review even when the operation completes."),
        "classify-audio":("Interpret all scores","Can more than one sound label be plausible in one window? Compare its audio with the complete Scores table. Reporting thresholds change flags, not the model’s fixed class vocabulary or captured scores."),
        "prebuilt-models":("Interpret the clusters","Find two similar customers and compare their assignments and segment profile. A cluster number is an arbitrary identifier. Without observed churn outcomes, why can it not be called a churn probability?"),
        "typed-decisions":("Inspect a declared answer","Compare one held-out ticket with its chosen option and all three option probabilities. With calibration off, an uncalibrated review status is expected. A high top-option probability is not proof that the decision is correct.")
    ]
}
