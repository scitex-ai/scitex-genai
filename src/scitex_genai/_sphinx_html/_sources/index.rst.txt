SciTeX GenAI
============

**scitex-genai** is the standalone home of generative-AI utilities — a
unified provider factory for LLMs today, with reserved namespaces for
agents, image, audio, video, embeddings, and multimodal as the field
fragments by modality.

.. code-block:: python

   from scitex_genai import GenAI

   ai = GenAI(model="gpt-4o-mini")
   print(ai("Explain neural networks in one sentence."))
   print("cost USD:", ai.cost)

The umbrella ``scitex-python`` exposes this package as ``scitex.genai``.
For classical / deep ML utilities (factored out of the same legacy
``scitex.ai``) see `scitex-ml
<https://github.com/ywatanabe1989/scitex-ml>`_.

Modality layout
---------------

============================  ============  ===================================
 Submodule                     Status        Notes
============================  ============  ===================================
``scitex_genai.llm``           implemented   Provider factory ``GenAI``.
``scitex_genai.agent``         reserved      claude-agent-sdk wrapper planned.
``scitex_genai.image``         reserved      Image generation / editing.
``scitex_genai.audio``         reserved      TTS / STT / music.
``scitex_genai.video``         reserved      Video generation.
``scitex_genai.embed``         reserved      Embeddings.
``scitex_genai.multimodal``    reserved      Any-to-any unified models.
============================  ============  ===================================

Reserved namespaces import successfully but raise ``NotImplementedError``
on attribute access — import paths are stable as features land.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
