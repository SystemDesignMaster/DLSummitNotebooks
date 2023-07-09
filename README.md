
# DLSummit Notebooks

DLSummit is an interactive system that systematically summarizes and visualizes the features learned by a deep learning model and how they interact for predictions. This repository contains the Python notebooks employed to generate the data used in the DLSummit visualization.

For the main DLSummit repository, please refer to [https://github.com/SystemDesignMaster/DLSummit][summit].

 ### Main notebooks:

* [`activation-matrices.ipynb`](activation-matrices.ipynb): generate aggregated activation matrices
* [`influence.py`](activation-matrices.ipynb): generate aggregated influence matrices
* [`activation-matrices-to-json.ipynb`](activation-matrices-to-json.ipynb): combine activation matrices per class into json format
* [`attribution-graph.ipynb`](dag.ipynb): generating class attribution graphs
* [`feature-vis-sprite-to-images.ipynb`](feature-vis-sprite-to-images.ipynb): split feature visualization sprites to single images

### Experimental notebooks:

* [`top-channels-used-per-layer.ipynb`](top-channels-used-per-layer.ipynb): analysis for determining which channels were used the most by all classes for all layers

## Live Demo

For a live demo, visit: [SystemDesignMaster.com/DLSummit][demo]

## Resources

We used the following ImageNet metadata:

* [https://github.com/google/inception/blob/master/synsets.txt](https://github.com/google/inception/blob/master/synsets.txt)
* [https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57](https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57)
* [https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)

## License

MIT License. See [`LICENSE.md`](LICENSE.md).

## Citation

```
@article{SystemDesignMaster2020DLSummit,
  title={DLSummit: Scaling Deep Learning Interpretability by Visualizing Activation and Attribution Summarizations},
  author={SystemDesignMaster},
  journal={IEEE Transactions on Visualization and Computer Graphics (TVCG)},
  year={2020},
  publisher={IEEE},
  url={https://SystemDesignMaster.com/DLSummit/}
}
```

## Contact

For questions or support [open an issue][issues] or contact [SystemDesignMaster][SystemDesignMaster].

[summit]: https://github.com/SystemDesignMaster/DLSummit
[SystemDesignMaster]: https://SystemDesignMaster.com
[demo]: https://SystemDesignMaster.com/DLSummit/
[issues]: https://github.com/SystemDesignMaster/DLSummitNotebooks/issues