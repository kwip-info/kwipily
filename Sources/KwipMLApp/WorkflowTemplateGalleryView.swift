import SwiftUI
import KwipMLCore

@MainActor final class WorkflowTemplateNavigation: ObservableObject {
    @Published var selectedID: String? = "sales-exceptions"
}

/// One navigation surface for welcome, blank canvas and menu entry points.
/// Creating a template always creates another project; no current canvas is replaced.
struct WorkflowTemplateGalleryView: View {
    @ObservedObject var navigation: WorkflowTemplateNavigation
    let create: (String) throws -> Void
    var close:()->Void = {}
    @State private var search = ""
    @State private var error: String?
    private let entries = (try? WorkflowTemplates.catalog()) ?? []
    private var filtered: [WorkflowTemplate] {
        entries.filter { search.isEmpty || ($0.title + " " + $0.category + " " + $0.summary).localizedCaseInsensitiveContains(search) }
    }
    private var selected: WorkflowTemplate? { entries.first { $0.id == navigation.selectedID } }
    var body: some View {
        VStack(spacing:0) {
            HStack {
                VStack(alignment:.leading,spacing:4) {
                    Text("Start from a template").font(.system(size:22,weight:.semibold))
                    Text("A real canvas you can inspect, edit and run with your data.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                Spacer()
                Button("Close",action:close).buttonStyle(FlatButtonStyle()).keyboardShortcut(.cancelAction)
                    .accessibilityIdentifier("template.gallery.close")
            }.padding(20)
            Divider()
            HSplitView {
                VStack(alignment:.leading,spacing:12) {
                    TextField("Find a workflow",text:$search).textFieldStyle(.plain).padding(9)
                        .background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
                        .accessibilityLabel("Find a workflow")
                    ThemedScrollView {
                        VStack(spacing:0) {
                            ForEach(filtered) { entry in
                                Button { navigation.selectedID = entry.id } label: {
                                    VStack(alignment:.leading,spacing:5) {
                                        Text(entry.category.uppercased()).font(.system(size:9,weight:.semibold)).foregroundStyle(MLTheme.muted)
                                        Text(entry.title).font(.system(size:13,weight:.medium)).frame(maxWidth:.infinity,alignment:.leading)
                                    }.padding(12).background(navigation.selectedID == entry.id ? MLTheme.rose.opacity(0.14) : Color.clear)
                                        .contentShape(Rectangle())
                                }.buttonStyle(.plain).accessibilityAddTraits(navigation.selectedID == entry.id ? .isSelected : [])
                                Divider()
                            }
                            if filtered.isEmpty { Text("No matching templates").font(.caption).foregroundStyle(MLTheme.muted).padding(16) }
                        }
                    }
                }.padding(16).frame(minWidth:250,idealWidth:290,maxWidth:340)
                ThemedScrollView {
                    if let entry = selected {
                        VStack(alignment:.leading,spacing:20) {
                            Text(entry.title).font(.system(size:24,weight:.semibold))
                            Text(entry.summary).font(.system(size:14)).foregroundStyle(MLTheme.muted)
                            InspectorSection("What you need",symbol:"checklist",identifier:"template.setup") {
                                ForEach(Array(entry.setup.enumerated()),id:\.offset) { index, text in
                                    HStack(alignment:.top,spacing:10) {
                                        Text(String(index+1)).font(.caption.monospacedDigit()).foregroundStyle(MLTheme.rose).frame(width:16)
                                        Text(text).font(.system(size:12)).fixedSize(horizontal:false,vertical:true)
                                    }
                                }
                            }
                            VStack(alignment:.leading,spacing:7) {
                                Text("Keep in mind").font(.system(size:12,weight:.semibold))
                                Text(entry.limitations).font(.system(size:12)).foregroundStyle(MLTheme.muted)
                            }
                            VStack(alignment:.leading,spacing:5) {
                                Text(entry.attribution).font(.caption)
                                if let url=URL(string:entry.sourceURL) { Link("Source and data requirements ↗",destination:url).font(.caption).foregroundStyle(MLTheme.rose) }
                                Text("Template v\(entry.version) · Independent project · No data or access permissions copied").font(.caption2).foregroundStyle(MLTheme.muted)
                            }
                        }.padding(24).frame(maxWidth:.infinity,alignment:.leading)
                    } else {
                        Text(entries.isEmpty ? "Templates could not be loaded. Your existing projects are unchanged." : "Choose a template to see its setup.")
                            .foregroundStyle(MLTheme.muted).padding(24)
                    }
                }.frame(minWidth:400)
            }
            Divider()
            HStack {
                Text(error ?? "Creates a new project. Set up your data in the next window.")
                    .font(.caption).foregroundStyle(error == nil ? MLTheme.muted : .red).lineLimit(2)
                Spacer()
                Button("Create new workflow") {
                    guard let selected else { return }
                    do { try create(selected.id); error=nil } catch { self.error="The template could not open. Your existing canvas is unchanged." }
                }.buttonStyle(FlatButtonStyle()).disabled(selected == nil)
                    .accessibilityIdentifier("template.create")
            }.padding(16)
        }.background(MLTheme.panel)
    }
}
