import SwiftUI

/// Flat, framed guidance. The existing lesson entry owns all lesson launch behavior.
struct LearningJourneyView: View {
    let search: String
    @Binding var selectedID: String?
    let openEntry: (String) -> Void
    private var journeys: [LearningJourney] { LearningJourney.matching(search) }
    private var selected: LearningJourney? { journeys.first { $0.id == selectedID } }
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Learning paths").font(.system(size: 24, weight: .semibold))
            Text("Six suggested journeys through the current lessons. Start where it helps you; there are no required gates.")
                .foregroundStyle(MLTheme.muted)
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(journeys) { journey in
                    LearningPathRow(journey: journey, selected: selectedID == journey.id) { selectedID = journey.id }
                }
            }
            if journeys.isEmpty {
                Label("No matching learning path", systemImage: "magnifyingglass").padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
            }
            if let journey = selected {
                framed(journey.title, symbol: "map") {
                    Text(journey.objective).fontWeight(.medium)
                    InspectorSection("Before you start", symbol: "info.circle", identifier: "learning.journey.prerequisites." + journey.id, initiallyExpanded: false) {
                        Text(journey.prerequisites).font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                    }
                    InspectorSection("Runtime and execution budget", symbol: "slider.horizontal.3", identifier: "learning.journey.runtime." + journey.id, initiallyExpanded: false) {
                        Text(journey.runtimeGuidance).font(.system(size: 12)).foregroundStyle(MLTheme.muted)
                    }
                    if let first = journey.steps.first {
                        Button("Start · " + first.title) { openEntry(first.entryID) }
                            .accessibilityIdentifier("learning.journey.start." + journey.id)
                    }
                }.id("learning.path.details")
                framed("Recommended order", symbol: "list.number") {
                    ForEach(Array(journey.steps.enumerated()), id: \.element.id) { index, step in
                        stepRow(step, number: String(index + 1), optional: step.isOptional)
                        if index + 1 < journey.steps.count { Divider() }
                    }
                }
                if !journey.optionalSteps.isEmpty {
                    framed("Optional detours", symbol: "arrow.triangle.branch") {
                        ForEach(journey.optionalSteps) { step in stepRow(step, number: "+", optional: true) }
                    }
                }
                framed("What to try next", symbol: "arrow.right") {
                    Text(journey.nextAction).font(.system(size: 13))
                    Text("Opening a lesson does not complete it. Guided checks are practice; verified assessment status comes from a current passed assessment result.")
                        .font(.caption).foregroundStyle(MLTheme.muted)
                }
            } else if !journeys.isEmpty {
                Text("Choose a journey to see its start, ordered stops and optional detours.").font(.callout).foregroundStyle(MLTheme.muted)
            }
        }
    }
    private func stepRow(_ step: LearningJourney.Step, number: String, optional: Bool) -> some View {
        HStack(alignment: .top, spacing: 12) {
            Text(number).font(.system(size: 13, weight: .semibold)).foregroundStyle(MLTheme.rose)
                .frame(width: 28, height: 28).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
            VStack(alignment: .leading, spacing: 5) {
                Text(step.title).font(.system(size: 13, weight: .semibold))
                Text(step.advice).font(.system(size: 12)).foregroundStyle(MLTheme.muted).fixedSize(horizontal: false, vertical: true)
                Text((optional ? "Optional · " : "") + step.evidence.rawValue).font(.system(size: 11)).foregroundStyle(MLTheme.muted)
            }.frame(maxWidth: .infinity, alignment: .leading)
            Button("Open") { openEntry(step.entryID) }.accessibilityLabel("Open " + step.title)
                .accessibilityIdentifier("learning.journey.step." + step.entryID)
        }.padding(.vertical, 6)
    }
    private func framed<Content: View>(_ title: String, symbol: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: symbol).font(.system(size: 15, weight: .semibold))
            Divider()
            content()
        }.padding(16).frame(maxWidth: .infinity, alignment: .leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
}
