// Offline CPU text recognition of frozen textbook page images; no network or labels.
import Foundation
import Vision

guard CommandLine.arguments.count > 1 else {
    fatalError("Supply one or more local page image paths")
}
for filename in CommandLine.arguments.dropFirst() {
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequestRevision3
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = false
    request.usesCPUOnly = true
    request.minimumTextHeight = 0.003
    let handler = VNImageRequestHandler(url: URL(fileURLWithPath: filename), options: [:])
    let started = Date()
    try handler.perform([request])
    let observations = (request.results ?? []).sorted {
        let a = Int(($0.boundingBox.midY * 1000).rounded())
        let b = Int(($1.boundingBox.midY * 1000).rounded())
        return a == b ? $0.boundingBox.minX < $1.boundingBox.minX : a > b
    }
    let lines: [[String: Any]] = observations.compactMap { observation in
        guard let candidate = observation.topCandidates(1).first else { return nil }
        let box = observation.boundingBox
        return ["text": candidate.string, "confidence": candidate.confidence,
                "box": [box.minX, box.minY, box.width, box.height]]
    }
    let result: [String: Any] = ["file": filename, "revision": request.revision,
        "cpu_only_requested": true, "language_correction": false,
        "wall_seconds": Date().timeIntervalSince(started), "lines": lines]
    let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
}
