import SwiftUI
import KwipMLCore

/// Model/affine adapter; all ribbon behavior is shared with data and reusable blocks.
struct ComponentPaletteView: View {
    @ObservedObject var store:WorkspaceStore
    var allowedTypes:Set<String>? = nil
    private var entries:[ComponentRibbonEntry] {
        var values = store.registry.components.filter {allowedTypes?.contains($0.id) ?? true}.map {
            ComponentRibbonEntry(id:$0.id,title:$0.title,shortTitle:ComponentStyle.shortTitle($0.id,fallback:$0.title),purpose:ComponentStyle.purpose($0.id),category:$0.category,typeIcon:$0.id)
        }
        if allowedTypes == nil && !store.isLearningLessonActive && store.tutorialStep == nil { values.append(ModelBlockPalette.entry) }
        return values
    }
    var body: some View {
        ComponentRibbonView(entries:entries,isBusy:store.isBusy,onAdd:{ type in
            guard !store.isBusy,entries.contains(where:{$0.id == type}) else {return}
            if type == ModelBlockPalette.type { let point=store.paletteInsertionPoint(type:type,start:CGPoint(x:60,y:420)); _ = store.addModelBlock(x:point.x,y:point.y) }
            else { let point=store.paletteInsertionPoint(type:type,start:CGPoint(x:60,y:420)); store.addNode(type:type,x:point.x,y:point.y) }
        },onDragStart:{type in store.statusMessage="Drag \(store.registry.component(type:type)?.title ?? type) onto the canvas."}, restricted: allowedTypes != nil)
    }
}
