import AppKit
import Foundation
import KwipMLCore

extension WorkspaceStore {
    func chooseCSV() {
        guard !isBusy else { return }
        let panel = NSOpenPanel(); panel.title = "Choose a local CSV dataset"; panel.allowsMultipleSelection = false; panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try sourceAccess.remember(url) }
        catch { dataError = "Could not retain source access: \(error.localizedDescription)"; return }
        setExecution { $0.preparedTraining = nil; $0.datasetCSV = url.path; $0.featureColumns = []; $0.targetColumn = nil }
        inspectData()
    }
    func useSampleData() { guard !isBusy else { return }; setExecution { $0.preparedTraining = nil; $0.datasetCSV = nil; $0.featureColumns = []; $0.targetColumn = nil }; dataPreviewSourcePath = nil; dataColumns = []; dataRows = []; dataError = nil; isInspectingData = false }
}
