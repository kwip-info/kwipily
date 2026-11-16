import Foundation
import KwipMLCore

@MainActor enum LessonContentCatalog {
    private static var cache:[String:LessonContent] = [:]
    private static let componentEntries=LearnCatalogContent.componentExampleEntries
    static func content(entryID:String,operation:String?=nil)->LessonContent? {
        if operation == nil,let cached=cache[entryID] {return cached}
        var content=make(entryID:entryID,operation:operation)
        let practiceAddresses:[String:WorkflowNodeAddress] = [
            "columns":ColumnsPracticeExample.columnsAddress,"filter":FilterPracticeExample.filterAddress,
            "file-input":FileInputViewPracticeExample.sourceAddress,"formula":FormulaPracticeExample.address,
            "preprocess":PreprocessPracticeExample.address,"partition":PartitionPracticeExample.address,
            "union":UnionPracticeExample.address,"summarize":SummarizePracticeExample.address,
            "reshapeTable":ReshapeTablePracticeExample.address,"splitText":.init(domain:.data,localID:"split"),
            "sort":.init(domain:.data,localID:"sort"),"unique":.init(domain:.data,localID:"unique"),
            "text-encoding":TextEncodingPracticeExample.address,"sequence-windows":SequenceWindowsPracticeExample.address]
        if let address=practiceAddresses[entryID],var annotated=content {
            annotated.steps=annotated.steps.map {step in var result=step;if result.nodeFocus == nil {result.nodeFocus = .init(addresses:[address])};return result}
            content=annotated
        }

        if let value=content {content=LessonCanvasReferences.annotate(FeatureLessonReview.refine(value))}
        if operation == nil,let content {cache[entryID]=content}
        return content
    }

    private static func make(entryID: String, operation: String? = nil) -> LessonContent? {
        guard let item=LearningLibraryPresentation.items.first(where:{$0.id==entryID}) else {return nil}
        func deck(_ steps:[LessonContentStep])->LessonContent {.init(entryID:entryID,title:item.title,kind:item.kind,steps:steps)}
        if entryID=="self-play" {return deck(SelfPlayLessonContent.steps)}
        if let steps=TrainingTopicLessonCopy.steps(entryID,operation:operation) {return deck(steps)}
        if let operation {
            if let steps=TensorTopicLessonCopy.steps(operation) {return deck(steps)}
            if item.kind=="Construction",let steps=LayerTopicLessonCopy.steps(operation) {return deck(steps)}
            if entryID=="module-composition-examples",let steps=LayerOverviewLessonCopy.workedSteps(operation) {return deck(steps)}
            let guide=LessonAudienceGuide.make(operation:operation)
            return deck([LessonContent.step("goal","What you are building",guide.goal)] + guide.steps.enumerated().map {LessonContent.step("build.\($0.offset)","Build · \($0.offset+1)",$0.element,kind:.canvasAction)} + [
                .init(id:"check",title:"Check your construction",blocks:[.init(id:"check.action",kind:.check,title:"",text:"Compare measured results with the task's requirements. A practice check is separate from an assessment."),.init(id:"check.terms",kind:.detail,title:"Terms",text:guide.terms.map {"**\($0.name)** — \($0.meaning)"}.joined(separator:"\n\n"))])])
        }
        if entryID=="model-output" {
            return deck(markdownSteps(ModelOutputPracticeExample.instructions).map {step in
                var result=step
                if step.title=="What Model Output does" {
                    result.blocks.append(.init(id:step.id+".canvas",kind:.canvasAction,title:"",text:ModelOutputPracticeExample.firstCanvasAction))
                    result.blocks.append(.init(id:step.id+".reflection",kind:.question,title:"Check your understanding",text:ModelOutputPracticeExample.understanding,answer:ModelOutputPracticeExample.understandingAnswer))
                } else if ["Follow the result","Try a deliberate mismatch","Practice independently"].contains(step.title) {
                    result.blocks[0].kind = .canvasAction
                }
                if !step.blocks.contains(where:{$0.text.hasPrefix("This is an unscored practice")}) {
                    result.nodeFocus = .init(addresses:step.title=="Follow the result" ? [ModelOutputPracticeExample.address,.init(domain:.model,localID:"classifier")]:[ModelOutputPracticeExample.address])
                }
                return result
            })
        }
        if entryID=="model-input" {
            return deck(ModelInputLessonContent.steps.map {step in
                var result=step;result.nodeFocus = .init(types:step.id=="follow" || step.id=="meaning" ? ["input","embedding"]:["input"]);return result
            })
        }
        if entryID=="guided" {
            return deck(GuidedLessonCopy.overview)
        }
        if let steps=LayerLegacyExampleLessonCopy.steps(entryID) {return deck(steps)}
        if let steps=LayerOverviewLessonCopy.steps(entryID) {return deck(steps)}
        if let steps=TensorTopicLessonCopy.exampleSteps(entryID) {return deck(steps)}
        if let entry=componentEntries.first(where:{$0.componentID==entryID}) {
            let learning=entry.learning
            if let steps=BundledExampleLessonCopy.steps(entryID,learning:learning) {return deck(steps)}
            return deck(([LessonContent.step("goal","What to explore",learning.subtitle)] + learning.paragraphs.enumerated().map {LessonContent.step("explain.\($0.offset)",learning.disclosureTitle,$0.element)} + [
                .init(id:"inspect",title:"Inspect the result",blocks:[.init(id:"inspect.body",kind:.observation,title:"",text:"Select the component and inspect its captured Output after running. Compare what entered with what left the component.",targetType:entryID),.init(id:"inspect.formula",kind:.visual,title:"Relationship",text:learning.formula,visual:.init(kind:.equation))])]).map {step in
                var result=step;result.nodeFocus = .init(types:[entryID]);return result
            })
        }
        if let lesson=try? DataEngineeringPractice.lesson(entryID) {
            return deck(authored(objective:lesson.objective,steps:lesson.steps,terms:lesson.terms,boundary:lesson.inspect+"\n\n"+lesson.disclosure).map {step in
                var result=step;result.nodeFocus = .init(types:[lesson.operatorID]);return result
            })
        }
        if let steps=DataExampleLessonCopy.steps(entryID) {return deck(steps)}
        if entryID=="data-preparation-course" {return deck(DataPreparationLessonContent.railSteps)}
        if let steps=DataPracticeLessonCopy.steps(entryID) {return deck(steps)}
        if let steps=authoredContent(entryID) {return deck(steps)}
        // Older family introductions remain accessible as reference/launch controls.
        // Generic guidance makes no unmeasured mathematical correctness claim.
        return deck([
            LessonContent.step("goal","What to explore",LearningLibraryPresentation.purpose(item)),
            LessonContent.step("build","Explore on the canvas","Use the lesson's exercise choices to open its supplied canvas. Select a component to inspect its configuration and connections. Change one setting at a time.",kind:.canvasAction),
            LessonContent.step("inspect","Read the evidence","Run the example or use Check construction where the lesson supports it. Inspect the measured output and diagnostics before drawing a conclusion.",kind:.observation)])
    }
    static func markdownSteps(_ markdown:String)->[LessonContentStep] {
        var result:[LessonContentStep]=[]
        var title="Explore";var chunks:[String]=[]
        func flush() {
            for (index,text) in chunks.enumerated() where !text.isEmpty {
                result.append(LessonContent.step("section."+title+".\(index)",title,text))
            }
            chunks=[]
        }
        for paragraph in markdown.components(separatedBy:"\n\n") {
            let lines=paragraph.components(separatedBy:"\n")
            if let first=lines.first,first.hasPrefix("**"),first.hasSuffix("**") {
                flush();title=String(first.dropFirst(2).dropLast(2));chunks.append(lines.dropFirst().joined(separator:"\n"))
            } else {chunks.append(paragraph)}
        }
        flush();return result
    }
    static func authored(objective:String,steps:[String],terms:String="",boundary:String="")->[LessonContentStep] {
        var result=[LessonContent.step("goal","What you are building",objective)]
        result += steps.enumerated().map {LessonContent.step("try.\($0.offset)","Try it · \($0.offset+1)",$0.element,kind:.canvasAction)}
        if !terms.isEmpty {result[0].blocks.append(.init(id:"goal.terms",kind:.detail,title:"Words to know",text:terms))}
        if !boundary.isEmpty {result.append(LessonContent.step("inspect","What to inspect",boundary,kind:.observation))}
        return result
    }
    private static func authoredContent(_ id:String)->[LessonContentStep]? {
        switch id {
        case "columns": return authored(objective:ColumnsLessonContent.objective,steps:ColumnsLessonContent.steps,terms:ColumnsLessonContent.terms,boundary:ColumnsLessonContent.boundary)
        case "filter": return authored(objective:FilterLessonContent.objective,steps:FilterLessonContent.steps,terms:FilterLessonContent.terms,boundary:FilterLessonContent.boundary)
        case "prebuilt-models":return authored(objective:PrebuiltModelsLessonContent.objective,steps:PrebuiltModelsLessonContent.steps,terms:PrebuiltModelsLessonContent.terms,boundary:PrebuiltModelsLessonContent.boundary)
        case "document-classification":return authored(objective:DocumentClassificationLessonContent.objective,steps:DocumentClassificationLessonContent.steps,terms:DocumentClassificationLessonContent.terms,boundary:DocumentClassificationLessonContent.boundary)
        case "game-episodes":return authored(objective:GameEpisodeLessonContent.objective,steps:GameEpisodeLessonContent.steps,terms:GameEpisodeLessonContent.terms,boundary:GameEpisodeLessonContent.boundary)
        case "game-rules":return authored(objective:GameRuleLessonContent.objective,steps:GameRuleLessonContent.steps,terms:GameRuleLessonContent.terms,boundary:GameRuleLessonContent.boundary)
        case "typed-decisions":return authored(objective:TypedDecisionLessonContent.objective,steps:TypedDecisionLessonContent.steps,terms:TypedDecisionLessonContent.terms,boundary:TypedDecisionLessonContent.boundary)
        case "classify-audio":return authored(objective:AudioClassificationLessonContent.objective,steps:AudioClassificationLessonContent.steps,terms:AudioClassificationLessonContent.terms,boundary:AudioClassificationLessonContent.boundary)
        case "transcribe-audio":return authored(objective:SpeechTranscriptionLessonContent.objective,steps:SpeechTranscriptionLessonContent.steps,terms:SpeechTranscriptionLessonContent.terms,boundary:SpeechTranscriptionLessonContent.boundary)
        case "extract-text":return authored(objective:OCRLessonContent.objective,steps:OCRLessonContent.steps,terms:OCRLessonContent.terms,boundary:OCRLessonContent.boundary)
        case "prepare-media": return authored(objective:PrepareMediaLessonContent.objective,steps:PrepareMediaLessonContent.steps,terms:PrepareMediaLessonContent.terms,boundary:PrepareMediaLessonContent.boundary)
        case "file-input": return authored(objective:FileInputLessonContent.objective,steps:FileInputLessonContent.steps,terms:FileInputLessonContent.terms,boundary:FileInputLessonContent.boundary)
        case "train-data-access": return authored(objective:TrainDataAccessLessonContent.objective,steps:TrainDataAccessLessonContent.steps,terms:TrainDataAccessLessonContent.terms,boundary:TrainDataAccessLessonContent.boundary)
        case "partition":return authored(objective:PartitionLessonContent.objective,steps:PartitionLessonContent.steps,terms:PartitionLessonContent.terms,boundary:PartitionLessonContent.boundary)
        case "preprocess":return authored(objective:PreprocessLessonContent.objective,steps:PreprocessLessonContent.steps,terms:PreprocessLessonContent.terms,boundary:PreprocessLessonContent.boundary)
        case "formula":return authored(objective:FormulaLessonContent.objective,steps:FormulaLessonContent.steps,terms:FormulaLessonContent.terms,boundary:FormulaLessonContent.boundary)
        case "splitText":return authored(objective:SplitTextLessonContent.objective,steps:SplitTextLessonContent.steps,boundary:SplitTextLessonContent.boundary)
        default:return nil
        }
    }
}
