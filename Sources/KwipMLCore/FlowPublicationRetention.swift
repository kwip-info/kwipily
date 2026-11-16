import Foundation
import Darwin

/// Retains pinned and interrupted authoring evidence; admission never guesses which files are disposable.
struct FlowPublicationRetention {
    var maximumFiles = 4096
    var maximumBytes = 256 * 1024 * 1024
    struct Full: LocalizedError {
        var errorDescription: String? {
            "This flow folder has reached its retained authoring-evidence limit. Raise the per-folder allowance in KwipML → Settings to retry. Inspect retained .kwipflow-draft-* and .kwipflow-publication-* records with recovery support; do not delete pinned files. Discarding the active draft remains available."
        }
    }
    func check(directory: Int32, incoming: [Data]) throws {
        let descriptor = openat(directory, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw FlowDocumentError.unsafeReference }
        guard let stream = fdopendir(descriptor) else { close(descriptor); throw FlowDocumentError.unsafeReference }
        defer { closedir(stream) }
        var files = incoming.count, bytes = incoming.reduce(0) { $0 + $1.count }, entries = 0
        guard files <= maximumFiles, bytes <= maximumBytes else { throw Full() }
        while let entry = readdir(stream) {
            entries += 1
            guard entries <= 100_000 else { throw FlowDocumentError.resourceLimit }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) }
            }
            guard name.hasPrefix(".kwipflow-draft-") || name.hasPrefix(".kwipflow-publication-") else { continue }
            var value = stat()
            guard fstatat(directory, name, &value, AT_SYMLINK_NOFOLLOW) == 0,
                  value.st_mode & S_IFMT == S_IFREG, value.st_size >= 0 else { throw FlowDocumentError.unsafeReference }
            files += 1
            guard files <= maximumFiles, value.st_size <= maximumBytes - bytes else { throw Full() }
            bytes += Int(value.st_size)
        }
    }
}
