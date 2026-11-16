import SwiftUI
import KwipMLCore

/// Policy-filtered data adapter; fixture sources never imply local file access.
struct DataFlowPaletteView: View {
    @ObservedObject var store:DataFlowWorkspaceStore
    private func category(_ type:String)->String {
        switch type {
        case "kwip.data.gameRules":return "Rules & simulation"
        case "kwip.data.sourceCSV","kwip.data.sourceFiles":return "Input"
        case "kwip.data.decide", "kwip.data.classifyAudio", "kwip.data.transcribeAudio", "kwip.data.extractText", "kwip.data.prepareMedia", "kwip.data.sequenceWindows", "kwip.data.textEncode","kwip.data.columns","kwip.data.formula","kwip.data.preprocess","kwip.data.splitText":return "Prepare"
        case "kwip.data.filterRows","kwip.data.sampleRows","kwip.data.sort","kwip.data.unique":return "Rows"
        case "kwip.data.partition":return "Partition"
        case "kwip.data.join","kwip.data.union","kwip.data.aggregate","kwip.data.reshapeTable":return "Combine & reshape"
        default:return "Inspect & output"
        }
    }
    private var entries:[ComponentRibbonEntry] {
        store.registry.components.map { definition in
            let fixture=store.policy == .normalizationLesson && definition.id == "kwip.data.sourceCSV"
            return ComponentRibbonEntry(id:definition.id,title:definition.title,shortTitle:fixture ? "Fixture source" : ComponentStyle.shortTitle(definition.id,fallback:definition.title),purpose:fixture ? "Adapter-owned observations for lesson fixtures" : ComponentStyle.purpose(definition.id),category:category(definition.id),typeIcon:definition.id)
        }
    }
    var body: some View {
        ComponentRibbonView(entries:entries,isBusy:store.isBusy,onAdd:{type in
            guard !store.isBusy,entries.contains(where:{$0.id == type}) else {return}
            let point=store.paletteInsertionPoint(type:type,start:CGPoint(x:60,y:450))
            store.addNode(type:type,x:point.x,y:point.y)
        },onDragStart:{type in store.statusMessage="Drag \(store.registry.component(type:type)?.title ?? type) onto the canvas."}, restricted: store.policy == .normalizationLesson)
    }
}
